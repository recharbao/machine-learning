
import numpy as np
def extract_data(filename,num_images,IMAGE_WIDTH):
    print('Extracting',filename)
    with open(filename,'rb') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH*IMAGE_WIDTH*num_images)
        data = np.frombuffer(buf,dtype=np.uint8).astype(np.float)
        data = data.reshape(num_images,IMAGE_WIDTH * IMAGE_WIDTH)
        return data




def extract_labels(filename,num_images):
    print('Extracting',filename)
    with open(filename,'rb') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf,dtype=np.uint8).astype(np.int64)
        return labels

