import matplotlib.pyplot as plt

# 模拟数据
edge_addition_rate = [10, 50, 100, 200, 300]
f1_scores_model1 = [41.77, 44.83, 44.51, 44.02, 42.98]
f1_scores_model2 = [40.21, 42.18, 42.01, 41.58, 40.97]
f1_scores_model3 = [44.22, 46.84, 46.23, 45.81, 44.56]
f1_scores_model4 = [38.97, 40.77, 40.76, 39.99, 40.02]
f1_scores_model5 = [27.06, 28.36, 28.01, 27.56, 27.44]
f1_scores_model6 = [22.78, 24.3, 23.98, 23.14, 23.01]
# f1_scores_model4 = [93.42,92.13,92.05,91.28,91.03,90.76,90.59,90.58,90.91,90.84]
#
# # 模拟误差范围
error_model1 = [0,0,0,0,0]
error_model2 = [0,0,0,0,0]
error_model3 = [0,0,0,0,0]
error_model4 = [0,0,0,0,0]
error_model5 = [0,0,0,0,0]
error_model6 = [0,0,0,0,0]
# error_model4 = [0.21,0.35,0.48,0.53,0.56,0.39,0.43,0.36,0.53,0.37]
#
# # 绘图
plt.figure(figsize=(12, 9))
#
# # 绘制折线
plt.plot(edge_addition_rate, f1_scores_model1, marker='o', color='blue', label='Cora')
plt.plot(edge_addition_rate, f1_scores_model2, marker='s', color='Orange', label='Citeseer')
plt.plot(edge_addition_rate, f1_scores_model3, marker='^', color='Green', label='Pubmed')
plt.plot(edge_addition_rate, f1_scores_model4, marker='+', color='red', label='Cornell')
plt.plot(edge_addition_rate, f1_scores_model5, marker='x', color='cyan', label='Chameleon')
plt.plot(edge_addition_rate, f1_scores_model6, marker='D', color='gray', label='Squirrel')
# plt.plot(edge_addition_rate, f1_scores_model4, marker='d', color='red', label='InfoMGF')
# #
# # # 填充浮动范围
# plt.fill_between(edge_addition_rate, [f1 - err for f1, err in zip(f1_scores_model1, error_model1)],
#                  [f1 + err for f1, err in zip(f1_scores_model1, error_model1)], color='blue', alpha=0.3)
# plt.fill_between(edge_addition_rate, [f1 - err for f1, err in zip(f1_scores_model2, error_model2)],
#                  [f1 + err for f1, err in zip(f1_scores_model2, error_model2)], color='Orange', alpha=0.3)
# plt.fill_between(edge_addition_rate, [f1 - err for f1, err in zip(f1_scores_model3, error_model3)],
#                  [f1 + err for f1, err in zip(f1_scores_model3, error_model3)], color='Green', alpha=0.3)
# #plt.fill_between(edge_addition_rate, [f1 - err for f1, err in zip(f1_scores_model4, error_model4)],
#  #                [f1 + err for f1, err in zip(f1_scores_model4, error_model4)], color='red', alpha=0.3)
#
plt.xlabel('$d$', fontsize=20)
plt.ylabel('Accuracy (%)', fontsize=20)  # 放大文字大小
plt.xticks(fontsize=18)  # 放大刻度文字大小
plt.yticks(fontsize=18)  # 放大刻度文字大小
plt.tick_params(axis='both', which='major', labelsize=18)  # 设置刻度大小
plt.xlim(0, 305)
plt.ylim(20, 50)

#plt.legend(loc='lower left', fontsize=14)
#添加图例，设置位置在图的下面
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=6, fontsize=14)

plt.savefig('/data/guoquanjiang/WS/MDGPT/model_node/plot.png', dpi=600)
plt.show()