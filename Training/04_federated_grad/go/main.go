package main

import (
	"context"
	"fmt"
	"log"
	"time"

	//"net"

	pb "federated_grad/federated_grad_pb"

	"github.com/petar/GoMNIST"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	NUM_EPOCHS    = 30
	BATCH_SIZE    = 32
	LEARNING_RATE = 0.01
)

type Coordinator struct {
	workers []pb.WorkerClient
	conns   []*grpc.ClientConn

	trainImages   *GoMNIST.Set
	globalWeights map[string]*pb.Weights
}

func NewCoordinator(workerAddrs []string) (*Coordinator, error) {
	c := &Coordinator{
		workers:       make([]pb.WorkerClient, len(workerAddrs)),
		conns:         make([]*grpc.ClientConn, len(workerAddrs)),
		globalWeights: make(map[string]*pb.Weights),
	}

	for i, addr := range workerAddrs {
		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			return nil, fmt.Errorf("error conectando a worker %s: %v", addr, err)
		}
		c.conns[i] = conn
		c.workers[i] = pb.NewWorkerClient(conn)
		log.Printf("✓ Conectado a worker %d: %s", i+1, addr)
	}

	log.Println("📚 Cargando dataset MNIST...")

	// Cargar training set
	train, _, err := GoMNIST.Load("../data/MNIST/raw")
	if err != nil {
		return nil, fmt.Errorf("error cargando MNIST: %v", err)
	}

	c.trainImages = train

	log.Printf("✓ Dataset cargado: %d imágenes de entrenamiento", train.Count())
	c.initializeGlobalWeights()
	return c, nil
}
func (c *Coordinator) initializeGlobalWeights() {
	// Inicializar pesos para fc.weight (784 x 10) y fc.bias (10)
	c.globalWeights["fc.weight"] = &pb.Weights{
		Values: make([]float32, 28*28*10),
		Shape:  []int32{10, 28 * 28},
	}
	c.globalWeights["fc.bias"] = &pb.Weights{
		Values: make([]float32, 10),
		Shape:  []int32{10},
	}

	log.Println("✓ Pesos globales inicializados")
}
func (c *Coordinator) Close() {
	for _, conn := range c.conns {
		conn.Close()
	}
}
func (c *Coordinator) GetBatch(startIdx, batchSize int) ([]float32, []int32) {
	endIdx := startIdx + batchSize
	if endIdx > c.trainImages.Count() {
		endIdx = c.trainImages.Count()
	}

	actualBatchSize := endIdx - startIdx

	// Preparar datos
	data := make([]float32, actualBatchSize*28*28)
	labels := make([]int32, actualBatchSize)

	for i := 0; i < actualBatchSize; i++ {
		idx := startIdx + i

		// Obtener imagen (es un []byte de 784 elementos)
		image, label := c.trainImages.Get(idx)

		// Convertir a float32 y normalizar [0, 1]
		for j := 0; j < 28*28; j++ {
			data[i*28*28+j] = float32(image[j]) / 255.0
		}

		labels[i] = int32(label)
	}

	return data, labels
}
func (c *Coordinator) broadcastWeights() error {
	for i, worker := range c.workers {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)

		_, err := worker.UpdateWeights(ctx, &pb.WeightRequest{
			Weights: c.globalWeights, // Usar el nombre correcto del proto
		})
		cancel()

		if err != nil {
			log.Printf("    Error actualizando pesos en worker %d: %v", i+1, err)
			return err
		}
	}
	return nil
}
func (c *Coordinator) aggregateGradients(gradientsList []map[string]*pb.Gradients) {
	numWorkers := float32(len(gradientsList))

	for paramName := range c.globalWeights {
		globalWeight := c.globalWeights[paramName]

		// Acumular gradientes de todos los workers
		for _, gradients := range gradientsList {
			if grad, exists := gradients[paramName]; exists {
				for i, gradVal := range grad.Values {
					// Aplicar gradiente descendente: weight = weight - lr * grad
					globalWeight.Values[i] -= LEARNING_RATE * gradVal / numWorkers
				}
			}
		}
	}
}
func (c *Coordinator) Train() {
	log.Println("\nIniciando entrenamiento distribuido")
	log.Printf("   Epochs: %d", NUM_EPOCHS)
	log.Printf("   Batch size: %d", BATCH_SIZE)
	log.Printf("   Workers: %d", len(c.workers))
	log.Printf("   Learning rate: %.4f\n", LEARNING_RATE)

	numBatches := c.trainImages.Count() / BATCH_SIZE

	for epoch := 0; epoch < NUM_EPOCHS; epoch++ {
		epochStart := time.Now()
		totalLoss := 0.0

		log.Printf("\n═══ Epoch %d/%d ═══", epoch+1, NUM_EPOCHS)

		firstIteration := (epoch == 0)
		batchesPerUpdate := len(c.workers)
		var accumulatedGradients []map[string]*pb.Gradients
		for batch := 0; batch < numBatches; batch++ {
			// Distribuir batch entre workers (round-robin)
			workerIdx := batch % len(c.workers)

			// Obtener datos del batch
			startIdx := batch * BATCH_SIZE
			data, labels := c.GetBatch(startIdx, BATCH_SIZE)

			// Enviar a worker
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			response, err := c.workers[workerIdx].TrainBatch(ctx, &pb.TrainRequest{
				Data:      data,
				Labels:    labels,
				BatchSize: int32(BATCH_SIZE),
			})
			cancel()

			if err != nil {
				log.Printf("Error en worker %d, batch %d: %v", workerIdx+1, batch, err)
				continue
			}

			totalLoss += response.Loss
			accumulatedGradients = append(accumulatedGradients, response.Gradients)

			if len(accumulatedGradients) >= batchesPerUpdate {
				// Solo en la primera vez, inicializa pesos desde gradientes
				if firstIteration && batch < batchesPerUpdate {
					log.Println("   Primera iteración: no actualizar aún, dejar pesos aleatorios del worker")
					accumulatedGradients = nil
					firstIteration = false
					continue
				}

				c.aggregateGradients(accumulatedGradients)
				accumulatedGradients = nil

				// Ahora sí, broadcast los nuevos pesos
				if err := c.broadcastWeights(); err != nil {
					log.Printf("Error broadcasting weights: %v", err)
				}
			}

			// Log cada 100 batches
			if (batch+1)%100 == 0 {
				avgLoss := totalLoss / float64(batch+1)
				log.Printf("   Batch %d/%d - Loss: %.4f", batch+1, numBatches, avgLoss)
			}

		}

		avgEpochLoss := totalLoss / float64(numBatches)
		epochDuration := time.Since(epochStart)

		log.Printf(" Epoch %d completada - Loss: %.4f - Duración: %s",
			epoch+1, avgEpochLoss, epochDuration)
	}

	log.Println("\nEntrenamiento finalizado")
}
func main() {
	// Lista de workers
	workerAddrs := []string{
		"localhost:50051",
		"localhost:50052",
	}

	coordinator, err := NewCoordinator(workerAddrs)
	if err != nil {
		log.Fatalf("Error inicializando coordinador: %v", err)
	}
	defer coordinator.Close()

	// Entrenar
	coordinator.Train()
}
