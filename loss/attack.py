import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# 对抗攻击
# FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + 2 * epsilon * sign_data_grad # 通常normalize到[0, 1]之间，这里normalize到[-1, 1]之间，因此增加转换系数 2 
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

# 高斯攻击函数
def gaussian_attack(image, epsilon):
    noise = torch.randn_like(image) * epsilon * 2 # 通常normalize到[0, 1]之间，这里normalize到[-1, 1]之间，因此增加转换系数 2
    perturbed_image = image + noise
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

# 对抗样本生成
def attack(model, device, testset, epsilon, attack_type='fgsm'):
    correct = 0
    adv_examples = []
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        #data = data.view(data.shape[0], -1)
        data.requires_grad = True

        inputdata = data
        inputdata = model(inputdata)
        output = inputdata
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        loss = nn.CrossEntropyLoss()(output, target)

        model.zero_grad()
        loss.backward()
        data_grad = data.grad
        #print(data_grad)

        if attack_type == 'fgsm':
            data_grad = data.grad
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack_type == 'gaussian':
            perturbed_data = gaussian_attack(data, epsilon)
        else:
            raise ValueError("Unknown attack type")
        
        output = perturbed_data

        output = model(output)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
        else:
            adv_examples.append((init_pred.item(), final_pred.item(), perturbed_data.squeeze().detach().cpu().numpy()))

    final_acc = correct / float(len(test_loader))
    return final_acc, adv_examples

if __name__ == "__main__":
    pass

