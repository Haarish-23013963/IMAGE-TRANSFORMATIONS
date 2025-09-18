# IMAGE-TRANSFORMATIONS
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import all the necessary modules
<br>

### Step2:
Choose an image and save it as filename.jpg
<br>

### Step3:
Use imread to read the image
<br>

### Step4:
Use cv2.warpPerspective(image,M,(cols,rows)) to translation the image
<br>

### Step5:
Use cv2.warpPerspective(image,M,(cols2,rows2)) to scale the image
<br>

## Program:

## Developed By: HAARISH V
## Register Number:212223230067

i)Image Translation
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('AOT.jpeg')
image.shape
plt.imshow(image[:,:,::-1])
plt.title('Original Image')
plt.show()

tx, ty = 100, 200  
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, M_translation, (636, 438))
plt.imshow(translated_image[:,:,::-1])
plt.title("Translated Image")
plt.axis('on')
plt.show()

```

ii) Image Scaling

```
# Image Scaling
fx, fy = 2.0, 1.0  
scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
plt.imshow(scaled_image[:,:,::-1]) 
plt.title("Scaled Image") 
plt.axis('on')
plt.show()

```
iii)Image shearing
```
# Image Shearing
shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  
sheared_image = cv2.warpAffine(image, shear_matrix, (636, 438))

plt.imshow(sheared_image[:,:,::-1])
plt.title("Sheared Image") 
plt.axis('on')
plt.show()
````
iv)Image Reflection
```
# Image Reflection
reflected_image = cv2.flip(image, 2)  # Flip the image horizontally (1 means horizontal flip)

# flip: 1 means horizontal flip, 0 would be vertical flip, -1 would flip both axes
# Show original image 
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis('off')

# Show reflected image 
plt.subplot(1, 2, 2)
plt.imshow(reflected_image[:,:,::-1])
plt.title("Reflected Image")
plt.axis('off')

plt.tight_layout()
plt.show()


```
v)Image Rotation
```

(height, width) = image.shape[:2] 
angle = 45  # Rotation angle in degrees (rotate by 45 degrees)
center = (width // 2, height // 2)  
M_rotation = cv2.getRotationMatrix2D(center, angle, 1) 

rotated_image = cv2.warpAffine(image, M_rotation, (width, height))  

plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))  
plt.title("Rotated Image")  
plt.axis('off')
```
vi)Image Cropping
```
# Image Cropping
x, y, w, h = 0, 0, 200, 150  

cropped_image = image[y:y+h, x:x+w]   # Format: image[start_row:end_row, start_col:end_col]
# Show original image 
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis('on')

# Show reflected image 
plt.subplot(1, 2, 2)
plt.imshow(cropped_image[:,:,::-1])
plt.title("Cropped Image")
plt.axis('on')

plt.tight_layout()
plt.show()


```
## Output:
### i)Image Translation
<img width="701" height="520" alt="image" src="https://github.com/user-attachments/assets/bcad4c4e-b576-476b-bce4-2fe877a799a5" />


### ii) Image Scaling
<img width="717" height="259" alt="image" src="https://github.com/user-attachments/assets/9dc1cb63-6197-43b3-af02-e81b6f694ec7" />

### iii)Image shearing
<img width="718" height="510" alt="image" src="https://github.com/user-attachments/assets/65701929-7d50-4ea3-954c-dd6481bba2ed" />

### iv)Image Reflection
<img width="1238" height="400" alt="image" src="https://github.com/user-attachments/assets/2bd4fe6f-0526-4fd6-a12b-859bfd1bb8f7" />

### v)Image Rotation
<img width="681" height="438" alt="image" src="https://github.com/user-attachments/assets/67646cd5-341f-428a-986a-0a6e7a5ba048" />

### vi)Image Cropping
<img width="1248" height="498" alt="image" src="https://github.com/user-attachments/assets/13c76b7c-4434-4a1b-bbe2-aa064bbbaf37" />

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
