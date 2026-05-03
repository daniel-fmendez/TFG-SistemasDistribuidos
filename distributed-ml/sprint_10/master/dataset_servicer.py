import grpc
import tarfile
import io
import dataset_pb2
import dataset_pb2_grpc

class DatasetServicer:
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.chunk_size = 4 * 1024 * 1024

    def DownloadDataset(self, request, context):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(self.path, arcname="dataset")
        
        total_size = buf.tell()
        buf.seek(0)
        chunk_index = 0

        while True:
            chunk = buf.read(self.chunk_size)
            if not chunk:
                break
            is_last = buf.tell() == total_size
            yield dataset_pb2.DatasetChunk(
                data=chunk,
                chunk_index=chunk_index,
                is_last=is_last
            )
            chunk_index += 1