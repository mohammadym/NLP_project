import matplotlib.pyplot as plt


most_occur ={'ali':3, 'mmd':4, 'reza':10}

most_common_news_values = []
most_common_news_keys = []
temp = 1
for item in most_occur:
    print(item)
    most_common_news_values.append(most_occur[item])
    most_common_news_keys.append(temp)
    temp += 1

plt.bar(most_common_news_keys, most_common_news_values)
plt.show()