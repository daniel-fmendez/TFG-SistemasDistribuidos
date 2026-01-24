package main

import (
	"context"
	"fmt"
	"log"
	"net"

	pb "federated-sum-grpc/federated_sum_pb"

	"google.golang.org/grpc"
)

type server struct {
	pb.UnimplementedProcessorServer
}

func (s *server) ComputeSum(ctx context.Context, in *pb.SumRequest) (*pb.SumReply, error) {
	fmt.Printf("Recibidos %d números para sumar\n", len(in.Numbers))

	var total int32 = 0
	for _, num := range in.Numbers {
		total += num
	}

	return &pb.SumReply{Result: total}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Error al escuchar: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterProcessorServer(s, &server{})

	fmt.Println("Sumador en GO listo en el puerto 50051")
	if err := s.Serve(lis); err != nil {
		log.Fatalf("Error al servir: %v", err)
	}
}
