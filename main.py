from data_util import read_train_data,read_test_data,prob_to_rles,mask_to_rle,resize,np
from model import get_unet
import pandas as pd

epochs = 50

# get train_data
train_img,train_mask = read_train_data()

# get test_data
test_img,test_img_sizes = read_test_data()

# get u_net model
u_net = get_unet()

# fit model on train_data
print("\nTraining...")
u_net.fit(train_img,train_mask,batch_size=16,epochs=epochs)

print("Predicting")
# Predict on test data
test_mask = u_net.predict(test_img,verbose=1)

# Create list of upsampled test masks
test_mask_upsampled = []
for i in range(len(test_mask)):
    test_mask_upsampled.append(resize(np.squeeze(test_mask[i]),
                                       (test_img_sizes[i][0],test_img_sizes[i][1]), 
                                       mode='constant', preserve_range=True))


test_ids,rles = mask_to_rle(test_mask_upsampled)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018.csv', index=False)

print("Data saved")