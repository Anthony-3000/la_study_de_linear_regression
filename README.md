# La Study De Linear Regression - [REPORT](##)

# Scholar Matric Prediction ( Final Score of Student )

> Data Head
![](./compiled/data_head.png)

```py
data = pd.read_csv("C:/Users/ROG/Desktop/machine learning/Student_Performance.csv").head(100)
print("\nShape of dataset:", data.shape)
print()
data.head()
```

![](./compiled/scatter_plot.png)

```py
X=data.iloc[:,2]
Y=data.iloc[:,-2]
fig, plot_data_set = plt.subplots()
plot_data_set.set_title("Single Feature DataSet")
plot_data_set.set_xlabel("Attendance Percentage")
plot_data_set.set_ylabel("Overall Score")
plot_data_set.scatter(X,Y,label="Data Points")
plt.show()
```

> Co Relationg Tabel

![](./compiled/corelation.png)

> HeatMap
![](./compiled/heatmap.png)

```py
plt.figure(figsize=(8,6))
sns.heatmap(cor_data, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")                                            #cmap='coolwarm' → color scheme
plt.show()                                                                                    #Blue = negative correlation (-1)
#White = zero correlation (0)
#Red = positive correlation (+1)
```

> Outliners
![](./compiled/outliers.png)
```py
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=data['Atten_Percent'])      #Atten_Percent--blue
sns.boxplot(x=data['Overall_Sc'])         #Overall_Sc--orange
plt.show()
```

> Histogram

![](./compiled/histogram.png)

```py
plt.hist(data["Atten_Percent"], bins=10)
plt.xlabel("Attendance Percentage")
plt.ylabel("frequency")
plt.title("Distribution of Attendance Percentage")
plt.show()
```

# Training The Moodeeell

# La Simple Single Feature Regression Model

![](./compiled/single_feature_training.gif)

```py
## our hyopthesis function is pre_y = w0+w1X

## intialize the value 

W0 = 0 ##assume this is the best fit line interscept
W1 = -5  # assume this is the best fit line slope
alpha = 0.001 ##learing rate
tolerance = 1e-6 ##tolerence
epoch = 0 ## checking how many time loop run

slope = [] ## storing each slop 
slope.append(W1)
intercept=[] ## storing each intercept
intercept.append(W0)

previous_cost = float('inf') 
cost_history = [] ##store the costinto this to plot further graph

m = len(Y)



while True:
    pre_Y = W0+W1*X ## predict new value based on that wo and w1

    cost = (1/(2*m))*np.sum(np.square(pre_Y-Y)) ## calculating the cost at that  w0 and w1
    cost_history.append(cost) ## storing the cost into cost history
    if abs(previous_cost - cost) <=tolerance: ## breaking the loop if diff is tending to zero
      break
    
    previous_cost = cost

    epoch= epoch+1

    ##updating the w0 and w1
    error = pre_Y-Y
    W0 = W0-(alpha/m)*(np.sum(error))   
    W1 = W1-(alpha/m)*(np.sum(X*error))
    slope.append(W1)
    intercept.append(W0)

print(f"Final W0: {W0}, Final W1: {W1}")
```

### Cost Function wrt to itterations

![](./compiled/simple_regression_cost_function.png)

# Effects of High Learning Rate

![](./compiled/regression_evolution_simple_with_high_learning_reate.gif)

# Effects of High Learning Rate

![](./compiled/regression_evolution_simple_with_low_learning_rate.gif)