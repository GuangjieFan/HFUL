import torch
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
import numpy as np

class ComplexMatrixMasker:
    def __init__(self, matrix, sparsity_ratio, device='cpu'):
        """
        初始化复杂矩阵掩码类
        """
        self.matrix = matrix.to(device)
        self.sparsity_ratio = sparsity_ratio
        self.device = device

    def eye_matrix(self):
        """
        生成相应单位阵
        """
        rows, cols = self.matrix.shape
        eye_mat = torch.eye(rows, device=self.device) if rows == cols else torch.eye(rows, cols, device=self.device)
        eye_mat0 = 1 - eye_mat
        return eye_mat, eye_mat0

    def avg_of_matrix(self, matrix):
        """
        计算传入矩阵的所有元素求和后的平均值
        """
        total_sum = torch.sum(torch.abs(matrix))
        num_elements = matrix.numel()
        average_value = total_sum / num_elements
       # print(average_value)
        return average_value

    def apply_gaussian_blur(self, matrix):
        """
        对掩码矩阵应用高斯模糊
        """
        matrix = matrix.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        kernel_size = 3
        sigma = 1.0

        # Create Gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma, self.device)
        kernel = kernel.expand(1, 1, -1, -1)  # Shape: [1, 1, kernel_size, kernel_size]

        blurred_mask = F.conv2d(matrix, kernel, padding=kernel_size // 2)
        blurred_mask = blurred_mask.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        return blurred_mask

    def create_gaussian_kernel(self, kernel_size, sigma, device):
        """
        创建高斯核
        """
        x = torch.arange(kernel_size, device=device) - kernel_size // 2
        x = x.float().pow(2)
        x = x.unsqueeze(0) + x.unsqueeze(1)
        kernel = torch.exp(-0.5 * x / sigma ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def apply_non_linear_transform(self, matrix):
        """
        对掩码矩阵应用非线性变换
        """
        non_linear_mask = torch.tanh(matrix * torch.pi * (2 * self.sparsity_ratio - 1))
        return non_linear_mask

    def apply_neighborhood_operations(self, matrix):
        """
        对掩码矩阵应用邻域操作
        """
        kernel = torch.tensor([[0.05, 0.3, 0.05],
                               [0.05, 0.3, 0.1],
                               [0.05, 0.3, 0.05]], device=self.device)

        matrix_reshaped = matrix.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        kernel_reshaped = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]

        neighborhood_mask = F.conv2d(matrix_reshaped, kernel_reshaped, padding=1).squeeze()

        return neighborhood_mask

    def adjust_sparsity(self, matrix):
        """
        调整掩码矩阵的稀疏性，使其稀疏度逼近矩阵平均值
        """
        average_value = self.avg_of_matrix(matrix)
        threshold = torch.quantile(matrix, average_value)
        sparse_mask = torch.where(matrix >= threshold, torch.tensor(1.0, device=self.device),
                                  torch.tensor(0.0, device=self.device))
        return sparse_mask

    def apply_complex_rules(self, matrix):
        """
        通过复杂规则对掩码矩阵进行处理
        """
        eye_matrix, eyematrix0 = self.eye_matrix()
        mask_eye_matrix = eye_matrix * self.matrix
        mask_eyematrix0 = eyematrix0 * self.matrix
        blurred_mask1 = self.apply_gaussian_blur(mask_eye_matrix)
        non_linear_mask1 = self.apply_non_linear_transform(blurred_mask1)
        neighborhood_mask1 = self.apply_neighborhood_operations(non_linear_mask1)
        complex_mask1 = self.adjust_sparsity(neighborhood_mask1)  # 这个是主对角线元素进行了掩码
        blurred_mask = self.apply_gaussian_blur(mask_eyematrix0)
        non_linear_mask = self.apply_non_linear_transform(blurred_mask)
        neighborhood_mask = self.apply_neighborhood_operations(non_linear_mask)
        complex_mask = self.adjust_sparsity(neighborhood_mask)#这个是除主对角线以外的元素进行了掩码
        return complex_mask1, complex_mask

    def add_matrix(self):
        """
        复杂矩阵掩码算法主函数
        """

        complex_mask1, complex_mask = self.apply_complex_rules(matrix=self.matrix)
        add_mask_matrx = complex_mask + complex_mask1
        result_mask_matrix = add_mask_matrx * self.matrix
        return result_mask_matrix / 8  # 假设客户端数为4

# if __name__ == "__main__":
#     # 创建输入矩阵
#     input_matrix = torch.tensor([
#     [0.759700227, -0.822952569, 0.101521343, -0.371874928, 0.804362833, 0.634739041, 0.442141235, -0.721707702, 0.745927691, -0.601792455],
#     [-0.830166757, 0.7596982, -0.471181422, 0.58390373, -0.910818338, -0.405422866, -0.667109668, 0.910231709, -0.62315309, 0.845341086],
#     [-0.090401918, -0.462979048, 0.752258527, -0.704299271, 0.469786167, -0.580899596, 0.589168608, -0.598586917, -0.258580536, -0.698162675],
#     [-0.444330037, 0.74718976, -0.739647985, 0.756455886, -0.754901052, 0.101900086, -0.871316552, 0.823374689, -0.430086255, 0.863755584],
#     [0.667008221, -0.904056907, 0.60223949, -0.629921973, 0.742750037, 0.200669408, 0.560279191, -0.874388516, 0.353303611, -0.81438446],
#     [0.788687229, -0.460705936, -0.351128906, 0.096006364, 0.405946374, 0.734597075, 0.033412427, -0.287460178, 0.696842074, -0.14514564],
#     [0.314832896, -0.649747789, 0.701315284, -0.851543486, 0.5734092, -0.19051674, 0.761376071, -0.760897398, 0.427371323, -0.830241799],
#     [-0.759365261, 0.950761318, -0.576103806, 0.695767641, -0.913314283, -0.250947088, -0.770328045, 0.756720471, -0.585519314, 0.901633382],
#     [0.784444511, -0.696075737, 0.13059482, -0.511065364, 0.621162593, 0.519617498, 0.658882737, -0.649294615, 0.737595963, -0.607748389],
#     [-0.636589229, 0.916521072, -0.715564251, 0.792235136, -0.857469082, -0.071333081, -0.862663507, 0.957803488, -0.511120617, 0.758727121]
# ])
#     sparsity_ratio = 0.55 # 设置sparsity_ratio值，0 < sparsity_ratio < 1
#
#     # 初始化复杂矩阵掩码类
#     masker = ComplexMatrixMasker(input_matrix, sparsity_ratio)
#
#     # 应用复杂矩阵掩码算法
#     result_matrix = masker.add_matrix()
#     print('result_mask:',result_matrix)
#     c25 = input_matrix*0.25
#     print("res:",result_matrix)
#     print("0.25c",c25)
#
#     print("差距differ_ratio :\n", c25-result_matrix)
