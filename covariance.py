import numpy as np
import matplotlib.pyplot as plt

# Check whether plots are interactive or not ... interactive mode is most useful from the command line
if plt.isinteractive() == True:
    print("Interactive mode is TRUE - exiting.")
    exit()
else:
    print("interactive mode is FALSE - remember to close graph window.")


# Generate 1000 samples from a 2D normal distribution
mean = [5.0, 8.0]
covariance_matrix = [[1, 0],
                     [0, 1]]

#1000 rows, 2 columns - each row a sample, each column a dimension
samples = np.random.multivariate_normal(mean, covariance_matrix, 1000)
#print(samples.shape)

#Compute the mean value over all samples in each dimension.
#Option 0 signifies that we want mean computed along first dimension (rows, i.e., samples)
actualmean = np.mean(samples,0)
print(actualmean)

# Plot scatter plot
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(samples[:,0], samples[:,1], marker=".", color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_box_aspect(1.0)
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])


#plt.show will halt thread - program cannot terminate until open windows have been closed
#plt.figure(figsize=(10,10))
plt.show()



exit()