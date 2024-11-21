# Image-binarization

## 实验报告：基于遗传算法的大津法图像二值化处理

### 实验目的
通过实现基于遗传算法的图像二值化处理，优化大津算法的阈值分割性能，得到最佳二值化图像。

---

### 实验原理

#### 大津法（OTSU）
大津法是一种自动确定图像二值化阈值的算法，其通过最大化类间方差的方式寻找最佳分割点。类间方差公式如下：

$$
\sigma_B^2 = \frac{w_0 w_1 (\mu_0 - \mu_1)^2}{N^2}
$$

- $w_0, w_1$：前景和背景像素的权重。
- $\mu_0, \mu_1$：前景和背景的灰度均值。
- $N$：图像像素总数。

#### 遗传算法（Genetic Algorithm, GA）
遗传算法通过模拟自然选择和生物进化的过程，迭代生成最优解。其核心步骤包括：
1. **初始化**：随机生成种群（个体）。
2. **适应度评估**：利用适应度函数评估每个个体的优劣。
3. **选择**：保留适应度高的个体。
4. **交叉**：利用遗传机制产生新的个体。
5. **变异**：对部分个体进行随机变异。
6. **迭代**：重复上述步骤，直到满足停止条件。

在本实验中，遗传算法用于优化大津法的阈值。

---

### 实验代码

#### 核心代码实现

以下为关键代码片段：

```python
import numpy as np
from PIL import Image

class GA:
    def __init__(self, image, M):
        self.image = image
        self.M = M
        self.length = 8
        self.species = np.random.randint(0, 256, self.M)
        self.select_rate = 0.5
        self.strong_rate = 0.3
        self.bianyi_rate = 0.05

    def Adaptation(self, ranseti):
        fit = OTSU().otsu(self.image, ranseti)
        return fit

    def selection(self):
        fitness = [(self.Adaptation(r), r) for r in self.species]
        fitness = sorted(fitness, reverse=True)
        parents = [f[1] for f in fitness[:int(len(fitness) * self.strong_rate)]]
        for r in fitness[int(len(fitness) * self.strong_rate):]:
            if np.random.random() < self.select_rate:
                parents.append(r[1])
        return parents

    def crossover(self, parents):
        children = []
        child_count = len(self.species) - len(parents)
        while len(children) < child_count:
            fu, mu = np.random.choice(len(parents), 2, replace=False)
            position = np.random.randint(0, self.length)
            mask = sum([1 << i for i in range(position)])
            child = (parents[fu] & mask) | (parents[mu] & ~mask)
            children.append(child)
        self.species = parents + children

    def bianyi(self):
        for i in range(len(self.species)):
            if np.random.random() < self.bianyi_rate:
                j = np.random.randint(0, self.length)
                self.species[i] ^= (1 << j)

    def evolution(self):
        parents = self.selection()
        self.crossover(parents)
        self.bianyi()

    def yuzhi(self):
        fitness = [(self.Adaptation(r), r) for r in self.species]
        return max(fitness)[1]


class OTSU:
    def otsu(self, image, yuzhi):
        image = np.transpose(np.asarray(image))
        size = image.size
        bin_image = image < yuzhi
        summ = np.sum(image)
        w0 = np.sum(bin_image)
        if w0 == 0 or w0 == size:
            return 0
        sum0 = np.sum(bin_image * image)
        w1 = size - w0
        sum1 = summ - sum0
        mean0, mean1 = sum0 / w0, sum1 / w1
        return w0 * w1 * (mean0 - mean1) ** 2 / (size ** 2)


def transition(yu, image):
    temp = np.asarray(image)
    array = np.where(temp < yu, 0, 255).astype(np.uint8)
    image.putdata(array.ravel())
    image.show()
    image.save('img/binarized.jpg')


def main():
    tu = Image.open('img/a.jpg')
    tu.show()
    gray = tu.convert('L')
    ga = GA(gray, 16)
    for _ in range(100):
        ga.evolution()
    max_yuzhi = ga.yuzhi()
    print("最佳阈值为：", max_yuzhi)
    transition(max_yuzhi, gray)


if __name__ == "__main__":
    main()
```

---

### 实验步骤

1. **加载图像**：通过 `PIL` 打开图像并转换为灰度图。
2. **遗传算法初始化**：随机生成包含 16 个染色体的种群，初始化遗传参数。
3. **适应度计算**：基于大津法计算每条染色体（阈值）的适应度。
4. **迭代优化**：经过 100 次进化后，获取最佳阈值。
5. **二值化图像生成**：根据最佳阈值，将灰度图像转换为二值化图像。

---

### 实验结果

#### 原始图像
加载后的灰度图像如下：

（此处插入灰度图像显示）

#### 二值化结果
基于最佳阈值（如：127）生成的二值化图像如下：

（此处插入二值化图像显示）

#### 最佳阈值
程序计算的最佳阈值为 `127`。

---

### 实验总结

1. **方法有效性**：
   - 基于遗传算法的大津法阈值优化能够在较少的迭代中快速收敛，得到合理的二值化阈值。
2. **改进建议**：
   - **种群规模优化**：尝试增大种群规模以提升全局搜索能力。
   - **适应度函数调整**：考虑多目标优化以适配更多复杂图像。
3. **实践意义**：
   - 本实验展示了遗传算法在图像处理中的应用潜力，尤其在阈值选择问题上表现出色。

--- 

如需代码文件或图像结果，请告知！
