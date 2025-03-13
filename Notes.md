# 1. KNN classifier
1. 适用于 数值型数据，对分类和回归都可以使用，但对高维数据效果较差，高维会维度灾难，高维空间中距离度量变得无意义。
1. Inductive learning: Use f: x -> y
2. Lazy Learning: Find nearest or most likelihood record
3. Nearest-Neighbor Classifiers
   1. How to define what is close:
![alt text](image.png)
   Euclidean distance and Manhattan Distance are correlated.
   cosine are used in face recognization: you know what you want to match, better deal with scale
   1 and 3 most frequently.
   2. Need rescale to make every attributes in the same scale, or there might exists some attributes affect more
   3. use encode when number doesn't means big or small, use 1,2,... when diff category have relationships such as 收入高中低
   4. Important to choose right k because small k leads to sensitive to noise and bigger k leads to include points from other classes
   5. Determine Class lable
      1. majority vote: every neighbor has the same impact on the classification
      2. weighted voting scheme: weight = 1/D(x*,xi)^2
   6. How to Solve Scaling:
      1. Normalization: Min-max
      2. Normalization: Standardization, z-score
   7. Training can let we know the weight and best K，训练时间几乎为 0（无训练过程），预测时间非常高，每次预测都要计算所有样本的距离。
# 2. Decision Tree 
1. For Classification, 适用于 数值型和类别型数据，可处理缺失值，适用于分类和回归任务
2. There could be more than one tree that fits the same data!
3. Decision Tree Induction: A tree can be learned by splitting training data into subsets based on outcomes of a feature test. This process is recursively applied on each derived subset until the subset at a node has all the same value of the
target variable or there is no improvement for prediction
   1. Hunt's Algorithm：找到能完全分类的变量进行分类
   2. CART
   3. ID#
   4. C4.5
   5. SLIQ
   6. SPRINT
4. Greedy strategy
5. Issues:
   1. Determine how to split:
      * how to specify the attribute test condition?
         * Attribute types: Continuous: Discretization or Binary Decision (compute intensive)
         * Number of ways to split(2-way, multi-way)
      * How to determine the best split?
         * Gini Index：smaller better
           Used in CART, SLIQ, SPRINT
         <img width="634" alt="image" src="https://github.com/user-attachments/assets/cdf18082-cb54-4445-a48c-986960273b86" />有几个类别就减多少个
         <img width="180" alt="image" src="https://github.com/user-attachments/assets/8a097331-9910-4e27-93e2-b2afdf4ea4e5" />

         * Entropy: smaller better
           <img width="550" alt="image" src="https://github.com/user-attachments/assets/bbf68512-e0ff-4e49-abe6-c3b7e42abd62" />
           <img width="515" alt="image" src="https://github.com/user-attachments/assets/e5905c66-3330-48e0-91fa-7f161db3bc39" />

         * Information Gain: larger better
           Used in ID3, C4.5. Tends to prefer splits that result in large number of
partitions, each being small but pure.父节点的熵减去子节点的熵
           <img width="642" alt="image" src="https://github.com/user-attachments/assets/2ce221b0-42e6-4f72-8db1-c5e81a4dc7d8" />
     
           衡量IG指标：GainRatio:<img width="709" alt="image" src="https://github.com/user-attachments/assets/33db41c6-6740-45a0-888e-ed2e106d8f54" />
信息增益的缺陷：IG 倾向选择取值多的特征（如身份证号）作为最优分裂特征，但这些特征实际上并不具备泛化能力。增益率的作用：通过分裂信息（SplitInfo）对信息增益进行归一化，避免 IG 偏向选择取值较多的特征。

         * Misclassification error，smaller better
           <img width="648" alt="image" src="https://github.com/user-attachments/assets/4bfcd695-35b1-428d-9b61-e8baf52a571e" />

   3. Determine when to stop splitting
      *  Stop expanding a node when all the records belong to the same class
      *  Stop expanding a node when all the records have similar attribute values
      *  Stop expanding a node when number of records is less than a threshold.
   4. Overfitting
      Overfitting results in decision trees that are more complex than necessary.
      Training error no longer provides a good estimate of how well the tree will perform on previously unseen records.
      Need new ways for estimating errors. Use generalization errors: Error on previously unseen testing set
   5. General Error
      * Occam's Razor: Definition: Given two models of similar generalization errors, one should prefer the simpler model over the more complex model

# 3. SVM

   1. Small margins 更容易受到Noise的影响。而且会有更高的testing error rate comparing with large margin.
