package main

import (
	"context"
	"log"
	"math/rand"
	"time"

	pb "hello-world-grpc/hello_pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var seededRand *rand.Rand = rand.New(
	rand.NewSource(time.Now().UnixNano()))

func StringWithCharset(length int, charset string) string {
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[seededRand.Intn(len(charset))]
	}
	return string(b)
}

const charset = "abcdefghijklmnopqrstuvwxyz" +
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

func main() {
	// 1. Establecemos la conexión con el servidor gRPC
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("No se pudo conectar: %v", err)
	}

	c := pb.NewHelloServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	r, err := c.SayHello(ctx, &pb.HelloRequest{Name: StringWithCharset(5, charset)})
	if err != nil {
		log.Fatal("Error al llamar: %v", err)
	}

	log.Printf("Respuesta del servidor: %s", r.GetMessage())

}
