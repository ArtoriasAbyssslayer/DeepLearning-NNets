# Auxiliary functions to move data to GPU or device in general
def to_device(data,device):
    "Move   tensors to chosen device"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be runnning on: ",device)
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

# The class below is Using Datalodaer to load the data to device with the use of above function
class DeviceDataLoader():
    "Wrap a dataloader to move data to a device"
    def __init__(self,dl,device) -> None:
        self.dl = dl
        self.device=device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        """Number of batches"""
        return len(self.dl)
        