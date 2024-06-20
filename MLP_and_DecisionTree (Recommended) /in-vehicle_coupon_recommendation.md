### 2. Data Exploration

The 'in-vehicle coupon recommendation Dataset' was gathered from a survey on Amazon Mechanical Turk. The survey asked about different driving situations like destination, presence of passengers, weather, and time of day to predict if drivers would accept coupons. 

The dataset includes 12,684 entries with 26 attributes or features.

C = Categorical

| Feature               | Description                                                                 | Data Type                      |
|-----------------------|-----------------------------------------------------------------------------|--------------------------------|
| destination           | Driver’s destination: "No Urgent Place", "Home", "Work"                      | C                              |
| passenger             | The passenger(s) in the car: "Alone", "Friend(s)", "Kid(s)", "Partner"       | C                              |
| weather               | The weather when the driver is driving: "Sunny", "Rainy", "Snowy"            | C                              |
| temperature           | The temperature when the driver is driving (in °F): "55", "80", "30"         | C (continuous data treated as categorical) |
| time                  | The time at which the driver is driving: "2PM", "10AM", "6PM", "7AM", "10PM" | C/numeric categorical (continuous data treated as categorical) |
| coupon                | Type of coupon that will be accepted: "Restaurant(<$20)", "Coffee House", "Carryout & Take away", "Bar, Restaurant($20-$50)" | C |
| expiration            | The expiration date of the coupon: "1d", "2h"                                 | C/numeric categorical (continuous data treated as categorical) |
| gender                | Driver’s gender: "Female", "Male"                                             | C                              |
| age                   | Driver’s age: "21", "46", "26", "31", "41", "50plus", "36", "below21"         | C/numeric categorical (continuous data treated as categorical) |
| maritalStatus         | Driver’s marital status: "Unmarried partner", "Single", "Married partner", "Divorced", "Widowed" | C |
| has_Children          | Whether the driver has child(ren) or not: 0: no, 1: yes                       | C (binary variable)            |
| education             | Driver’s educational background: "Some college - no degree", "Bachelors degree", "Associates degree", "High School Graduate", "Graduate degree (Masters or Doctorate)", "Some High School" | C |
| occupation            | Driver’s occupation: "Unemployed", "Architecture & Engineering", "Student", "Education&Training&Library", "Healthcare Support", "Healthcare Practitioners & Technical", "Sales & Related", "Management", "Arts Design Entertainment Sports & Media", "Computer & Mathematical", "Life Physical Social Science", "Personal Care & Service", "Community & Social Services", "Office & Administrative Support", "Construction & Extraction", "Legal", "Retired", "Installation Maintenance & Repair", "Transportation & Material Moving", "Business & Financial", "Protective Service", "Food Preparation & Serving Related", "Production Occupations", "Building & Grounds Cleaning & Maintenance", "Farming Fishing & Forestry" | C |
| income                | Driver’s income: "$37500 - $49999", "$62500 - $74999", "$12500 - $24999", "$75000 - $87499", "$50000 - $62499", "$25000 - $37499", "$100000 or More", "$87500 - $99999", "Less than $12500" | C/numeric categorical (continuous data treated as categorical) |
| Car                   | Car model driven by the driver: "Scooter and motorcycle", "crossover", "Mazda5" | C                              |
| Bar                   | The frequency of restaurant visits per month: "never", "less1", "13", "gt8", "nan48" | C                              |
| CoffeeHouse           | Frequency of cafe visits per month: "never", "less1", "48", "13", "gt8", "nan"  | C/numeric categorical (continuous data treated as categorical) |
| CarryAway             | Frequency of takeaway food consumption per month: "n48", "13", "gt8", "less1", "never" | C/numeric categorical (continuous data treated as categorical) |
| RestaurantLessThan20  | Frequency of restaurant visits per month, where the average expense per person is less than $20: "48", "13", "less1", "gt8", "never" | C/numeric categorical (continuous data treated as categorical) |
| Restaurant20To50      | Frequency of restaurant visits per month, where the average expense per person is between $20-$50: "13", "less1", "never", "gt8", "48", "nan" | C/numeric categorical (continuous data treated as categorical) |
| toCoupon_GEQ5min      | Open to travelling beyond a 5-minute distance to use the coupon: 0: no, 1: yes | C (binary variable)            |
| toCoupon_GEQ15min     | Open to travelling beyond a 15-minute distance to use the coupon: 0: no, 1: yes | C (binary variable)            |
| toCoupon_GEQ25min     | Open to travelling beyond a 25-minute distance to use the coupon: 0: no, 1: yes | C (binary variable)            |
| direction_same        | Whether the restaurant or cafe mentioned in the coupon is in the same direction as drivers’ current destination: 0: no, 1: yes | C (binary variable)            |
| direction_opp         | Whether the restaurant or cafe mentioned in the coupon is in the opposite direction as drivers’ current destination: 0: no, 1: yes | C (binary variable)            |
| Y                     | Whether the driver will accept the coupon or not: 0: no, 1: yes              | C (binary variable)            |

<br>

Missing values and data types:

6 features are integers (car, Bar, CoffeeHouse, CarryAway, RestarantLessThan20, and Restarant20to50). The remaining ones are objects.

<br>

#### Missing values:

The feature ‘car’ has the highest missing value rate of 99% (12576). This has been dropped as it can cause bias in the analysis. 

The remaining 5 features, “Bar”, “CoffeeHouse”, “CarryAway”, “RestaurantlessThan20”, and “Restaurant 20To50” have an average missing value rate of 1.25%. As per general guidelines, missing values that are less 5% have minimal impact on the overall analysis, therefore we will retain them

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/3dab7bfb-bb35-4c2f-a6d5-615d0c0aa497)

<br>

#### Duplicates:

74 duplicates have been removed. 

It is important to keep in mind that this dataset lacks a unique identifier or key that distinguishes individual records (such as a customer ID or transaction ID). There is no definitive way to determine if identical rows represent the same entity. It could be different labels assigned to the same record, or simply just a systematic error.

<br>

Visualisation / Analysis
---

#### Distribution of coupons:
![image](https://github.com/PixieParksie/UniProjects/assets/106667881/2d088c86-086c-4752-9e35-f7fcf5266ace) ![image](https://github.com/PixieParksie/UniProjects/assets/106667881/4e48c00a-3090-4f1f-8c2c-954264532644)



- Coffee House distributes the most coupons. It shows the highest acceptance rate but also experiences the highest rejection rate. 
- Coupons for light meals like restaurants(<20 seats) and takeaways tend to have higher acceptance rates than rejection rates, whereas coupons that have the lowest distribution rate like bars and restaurants (20-50 seats) typically face higher rejection rates.

Let’s investigate the factors that influence coupon type and acceptance rates in the following sequence: gender / occupation & Salary, marital status, situation, and context.

<br>

#### 1. Gender / Occupation & Salary:

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/7a3736c9-bd0c-4c15-884b-6daee649022e)

Both males and females have a similar acceptance rate. A larger proportion of females tend to accept the coupon compared to males.

<br>

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/5108f5d2-2ddc-4c8c-a9d6-de3061008c33)

There is a high ratio of drivers who fall under the categories of Unemployed, Student, Education&Training&Library, Sales & Related, and Computer & Mathematical occupations. 

<br>

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/c11100c1-9f8e-428c-9286-395871061f33)

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/36c72bcf-fa62-4ebc-a1db-6d7b2ad15848)

Categories that have higher rejection rates are Community/Social Service, Legal, and Retired.

<br>

#### 2. Marital Status:

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/425d7fb6-e047-4fc2-8b4a-17e04b14c349)
![image](https://github.com/PixieParksie/UniProjects/assets/106667881/a9a3bde5-245a-45b9-a441-3a92d0574b8e)

Coupons are distributed widely among drivers classified as 'single' and 'married partners' across all categories.

<br>

#### 4.	Contextual: TO BE EDITED

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/e2057418-7361-4d00-9870-fd09f32b85f2)

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/e2554ff4-fb9d-4fb3-8f54-defb02f2bdf1) ![image](https://github.com/PixieParksie/UniProjects/assets/106667881/dd997e0f-1e28-4bd2-ab65-e664bc4db7a1)

Drivers tend to accept the coupon more often when they are not in a hurry to reach a destination urgently. Acceptance rates are higher outside of rush hours, with a peak around 2pm, after lunchtime. Less popular times for accepting coupons are around 7am, during morning commute hours, and at 10pm, closer to bedtime.

<br>

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/ef29e9c6-f663-413b-a2a9-a7780a0c7379) ![image](https://github.com/PixieParksie/UniProjects/assets/106667881/867d2013-e069-4be1-aa26-2c3fd0d2c423)

Coupons are more frequently distributed on sunny days compared to cold and rainy days. People are more likely to reject coupons when the weather is rainy or cold.

<br>

#### 5. Situational:

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/2e0c1efc-d0da-4c96-a14f-7620a3c1beb3)

Acceptance rates are higher when drivers are offered coupons with longer expiration dates.

<br>

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/8056f6e9-fdee-4994-a86b-c08a446f24fb)

Drivers tend to decline the coupon as the distance to drive grows longer

<br>

Model one: Decision Tree Classifier:
---

Baseline model: used default parameters, resulting in a decision tree consisting of 4667 nodes and an accuracy score of 0.69 (2dp). 

#### Hyperparameter Tuning: 

10-fold CV scores for adjusted ‘max_depth’ parameter:

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/c06ffe2c-2363-4104-bd1e-1b31f5f5b394)
- Adjusted the max_depth parameter to limit the depth of the decision tree, testing different values of max_depth from 1 to 14 using 10-fold cross-validation for each value.
- Diagram below displays the average scores from 10-fold cv for various max_depth values. The optimal max_depth is 6.
- After adjusting the max_depth parameter, nodes in the tree decreased to 127, and the accuracy score improved to 0.70 (2dp)

<br>

#### Refinement:

10-fold CV scores for adjusted ‘max_leaf_nodes’ parameter

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/d5471d10-d2f9-4dc2-a947-31868744e831)
- I focused on adjusting the max_leaf_nodes parameter, as it limits the max number of leaf nodes in the decision tree.
- Used max_depth value of 6 as a starting point, then examined different max_leaf_node values from 2 to 39. Through this process, we determined that a max_leaf_node value of 22 resulted in the highest score from our 10-fold cross-validation.
- The model achieved an accuracy score of 0.70 (2dp). Adjustments did not significantly improve the accuracy of DTC.

<br>

#### Final optimised classification tree:

![DTC](https://github.com/PixieParksie/UniProjects/assets/106667881/246ae68f-359a-4c61-9507-d13b35b0d0aa)
- This binary tree comprises a total of 47 nodes - 24 internal nodes with 22 leaf nodes that make final classification decisions.
- The optimised tree has a depth of 6 levels. This depth value was identified as optimal during the decision tree's parameter optimisation (depth indicates the longest path from the initial node to any leaf node in the tree).

<br>

####  Model Evaluations:

Confusion Matrix

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/c5ffb783-741b-416f-b6ea-4812754e76ea)
- 1661 instances were correctly classified as 'accept' (TP).
- 970 instances were correctly classified as 'reject' (TN).
- 684 instances were incorrectly classified as 'accept' when they were actually 'reject' (FP).
- 468 instances were incorrectly classified as 'reject' when they were actually 'accept' (FN).

<br>

Model summary report

![image](https://github.com/PixieParksie/UniProjects/assets/106667881/e2e8c101-4de8-456a-b7b5-43f9ee0aa5cd)

Precision 0: 67% chance that the model will predict a driver will reject the coupon, it is accurate most of the time.

Precision 1: 71% chance that the model will predict a driver will accept the coupon, it is correct most of the time.

Recall 0: 59% chance that the model effectively captures most of the drivers who truly reject the coupon.

Recall 1: 78% chance that the model effectively captures most of the drivers who truly accept the coupon.

F1-score 0: Out of all instances predicted as 'reject' by the model, 63% are actually 'accept'.

F1-score 1: Out of all instances predicted as 'accept' by the model, 74% are actually 'accept'.

Accuracy: 70% of the predictions made by the model are correct (predictions across both 'accept' and 'reject' classes).


<br>

Evaluation: 

The model performs adequately in predicting coupon acceptance and rejection. Error rate of 30% indicates there are still opportunities for enhancement, especially in reducing false predictions. Further optimisation could focus on improving precision for 'reject' predictions and reducing overall prediction errors.
