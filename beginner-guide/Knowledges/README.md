# 模型篇

1. 寫出cross entropy的公式
> <img src="./cross-entropy-formula.png" width="200">
2. 解釋SGD
> 一optimizer
3. 解釋Adam，比較其與SGD的差異
> 一optimizer，效果較SGD好，故少使用SGD
4. 解釋ResNet主要解決的問題及為何有效
5. 解釋InceptionV3中，1x1 convolution的用意
6. VGG要做出熱力圖有個重要條件，請問為何
7. leaky relu主要想解決什麼樣的問題
8. 寫出softmax的公式及其微分
9. 解釋embedding
10. 使用embedding的時機為何
> 將圖片(元件)資訊帶進訓練模型中
11. 解釋batch normalization
> 在放入圖片時都會先normalization，故希望在每一層訓練後，將其數值normalization和原input一樣數值範圍
12. 除了batch normalization，再舉出三個normalization的方法
13. batch size增大learning rate也須成比例放大的依據為何
> 按邏輯，batch size大，每次迭代可看到的資訊也多，使用大learning rate容易得到好的答案，而batch size小，使用小learning rate以防其跑離最佳區間，亦或者跑不到最佳區間
14. 呈上，你覺得這是否合理
15. 列舉時間序列數據的特點
16. 解釋FGSM
17. 除了FGSM之外，再舉出一種white box attack的代表
18. 說明adversarial training的原理
> 利用較多雜訊的圖攻擊模型，以讓模型應付更general的圖片
19. 說明graph regularization的原理
20. 寫下使用graph regularization進行模型訓練時的完整流程
21. 列出五個使用regularization的方法，不限論文，可以是自己的想法
22. 寫出style transfer所使用的loss
23. 如果要將style transfer訓練改為使用regularization，說說你會怎麼做
24. 試舉出使用graph的缺點
25. 使用mixed precision有什麼需要主要的地方
> 會加速訓練，但準確率會相對比沒mixed還差
26. 簡單解釋tensorflow mirrored strategy使用何種方法合併模型參數
