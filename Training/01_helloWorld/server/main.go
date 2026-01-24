package main

import (
	"context"
	"log"
	"net"

	pb "hello-world-grpc/hello_pb"

	"google.golang.org/grpc"
)

type server struct {
	pb.UnimplementedHelloServiceServer
}

func (s *server) SayHello(ctx context.Context, req *pb.HelloRequest) (*pb.HelloResponse, error) {
	log.Printf("Received: %v", req.GetName())
	return &pb.HelloResponse{Message: "Hello " + req.GetName()}, nil
}

func main() {
	// 3. Escuchamos en el puerto 50051
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Error al escuchar: %v", err)
	}

	// 4. Creamos el servidor gRPC y lo registramos
	s := grpc.NewServer()
	pb.RegisterHelloServiceServer(s, &server{})

	log.Printf("Servidor corriendo en %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("Error al servir: %v", err)
	}
}
