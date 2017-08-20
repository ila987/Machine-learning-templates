#RE-INFORCEMENT LEARNING
#THOMPSON SAMPLING 

#import the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

#Implement T.S.
N = 10000
d = 10
numbers_of_reward_1 = integer(d)
numbers_of_reward_0 = integer(d)
ads_selected = integer(0)
total_reward = 0 
#loop over all users
for(n in 1:N){
  ad = 0 
  max_random =0
  #loop over all versions of the ads
  for (i in 1:d){
    random_beta = rbeta(n= 1, 
                        shape1= numbers_of_reward_1[i] + 1, 
                        shape2 = numbers_of_reward_0[i]+1)
    
    if(random_beta > max_random){
      max_random = random_beta
      ad = i 
    }
  }
  ads_selected = append(ads_selected,ad)
  reward = dataset [n, ad]
  if(reward ==1)
  {
    numbers_of_reward_1[ad] = numbers_of_reward_1[ad]+1
  }else{
    numbers_of_reward_0[ad] = numbers_of_reward_0[ad]+1
    
  }
}


#visualizing results with an histogram
hist(ads_selected, col= 'blue', main = 'Histogram of ads selection', xlab = 'Ads', ylab = '# of time each time was selected')