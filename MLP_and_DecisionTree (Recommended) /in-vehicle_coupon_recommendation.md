
### 2. Data Exploration

The 'in-vehicle coupon recommendation Dataset' was gathered from a survey on Amazon Mechanical Turk. The survey asked about different driving situations like destination, presence of passengers, weather, and time of day to predict if drivers would accept coupons. The dataset includes 12,684 entries with 26 attributes or features.


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
