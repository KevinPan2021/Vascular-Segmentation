from read_write import Load_Model
import torch

with torch.no_grad():
 
    device = 'cuda:1'
    load_model = Load_Model(name='IPNV2_msa', device=device)
    model = load_model.get()

    # model = model.to(device)

    volumes = torch.rand((1, 2, 128, 256, 256)).to(device)
    pms = torch.rand((1, 2, 256, 256)).to(device)
    resolution = torch.tensor([0, 1, 0], dtype=torch.float32).to(device)
    test, test2 = model(volumes, pms, resolution)
    print(test.shape)



# net = 