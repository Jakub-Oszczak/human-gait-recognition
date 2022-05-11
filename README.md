# human-gait-recognition

I wrote an AI program that takes data such as acceleration and angular velocity (gyro) from feet-mounted IMU and finds out if the person was walking or running.
While making this project I've improved my skills in data processing and AI.
The program first retrieves data from txt files, then delets offset, scales data. Then it searches for sections when feet isn't moving, zeroes those sections and then splits 
data into single steps. Next it removes first and last step, because it's often disturbed and also removes those steps that are significantly shorter or longer than mean, because those are also some incorrect steps.
Lastly it takes a mean from each step and passes it to AI algorithm which determines if it was a walk or run.

Data used in this program comes from user romanchereshnev, https://github.com/romanchereshnev/HuGaDB
