# Comprobar version
protoc-gen-go --version

# Coversion de .proto a GO
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative test.proto

# --go_out=. Indica al generador que el directorio de salida es el actual
# --go_opt=paths=source_relative Esta opción le dice al compilador que el archivo generado debe colocarse en la misma carpeta donde se encuentra el archivo .proto
# --go-grpc_out=. Le dice al plugin de gRPC que gener codigo para cliente/servidor en el directorio actual
# --go-grpc_opt=paths=source_relative Le dice que el codigo generado se quede en la carpeta del archivo fuente

# Para python3
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. test.proto
