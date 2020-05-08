#!/usr/bin/env python
# coding: utf-8

# ## Heat flow estimation from aerial radiometric measurements.
# 
# In this notebook we are concerned with the image processing of radiometric maps from Nova Scotia https://novascotia.ca/natr/meb/download/dp163.asp and the subsequent estimate of heat flow following the equations in Beamish and Busby (2016).
# 
# ### About the data
# 
# This data set consists of 7 JPEG images of radiometric data for the province of Nova Scotia. They include images showing Potassium (K, %), equivalent Thorium (eTh, ppm), equivalent Uranium (eU, ppm), the ratio Thorium/Potassium (eTh/K, ppm/%), the ratio Uranium/Potassium (eU/K, ppm/%), the ratio Uranium/Thorium (eU/eTh) and the Total Count at a 50m resolution. The images were created by combining radiometrics data provided by the Geological Survey of Canada (GSC) from their surveys flown at 1 km line-spacing across the entire province, and 7 detailed surveys flown at 250 m line spacing by the GSC in the following areas: East Kemptville, Liscomb, Ship Harbor, Gibraltor Hill, Granite Lake, Big Indian Lake, and Tantallon Lake. The images were produced by contractor M. S. King using funds provided under the Natural Resources Canada and Nova Scotia Department of Natural Resources joint project 'Mapping, Mineral and Energy Resource Evaluation, Central Nova Scotia', part of Natural Resources Canada's Targeted Geoscience Initiative (TGI-2) 2003-2005.
# 
# ### Image processing
# 
# Unfortunately there are two problems with the data as delivered. The first is the images have had a hillshading effect applied obscuring data values from a pure radiometric measurement. The second issue is the images don't have a colorscale, so it's impossible to convert the colours back into physical quantities without using some external heuristic or calibration. 
# 
# We deal with the first issue by converting the R,G,B channels in the image to H, S, V, and then dripping the V channel – which is where the hillshade effect lies. This is following the code from Matt Hall in a Gist called, Ripping data from pseudocolour images:
# 
# https://gist.github.com/kwinkunks/485190adcf3239341d8bebac94de3a2b#file-rip-data-2-py
# 
# (notebook saved without cell outputs because rendered outputs are too large)
#     

# In[ ]:


"""
If the colourmap matches all or part of the colour wheel or hue circle,
we can decompose the image to HSV and use H as a proxy for the data.
"""
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from skimage.color import rgb2hsv
import glob


# In[ ]:


def heat_equation(Cu, Cth, Ck, density=2700, c1=9.52, c2=2.56, c3=3.48):
    """
    Heat production equation from Beamish and Busby (2016)
    density is the density of the 
    
    density: rock density in kg/m3
    Cu: weight of uranium in ppm
    Cth: weight of thorium in ppm
    Ck: weight of potassium in %
    
    Returns: Radioactive heat production in W/m3
    """
    return (10e-5)*density*(c1 * Cu + c2 * Cth + c3 * Ck)
    


# In[ ]:


def heat_equation_no_density(Cu, Cth, Ck, c1=0.26, c2=0.07, c3=0.10):
    """
    Heat production equation from Beamish and Busby (2016)
    density is the density of the 
    
    Cu: weight of uranium in ppm
    Cth: weight of thorium in ppm
    Ck: weight of potassium in %
    
    Returns: Radioactive heat production in W/m3
    """
    return c1 * Cu + c2 * Cth + c3 * Ck


# In[ ]:


# Get the 7 radiometric maps and their names
fnames = glob.glob('i163nsaa_NS_Radiometric_Images_50m/jpg/*.jpg')
names = [fname.split('/')[-1].split('.')[0] for fname in fnames]


# In[ ]:


names


# In[ ]:


fnames


# In[ ]:


print(fname, name)
img = Image.open(fnames[1])


# In[ ]:


# Read the image and transform to HSV and save fig.
for fname, name in zip(fnames, names):
    print(fname, name)
    img = Image.open(fname)
    img_size = img.size
    rgb_im = np.asarray(img)[..., :3] / 255.
    hsv_im = rgb2hsv(rgb_im)
    hue = hsv_im[..., 0]
    # val = hsv_im[..., 2]
    # Make a new figure.
    my_dpi = 96
    plt.figure(figsize=(img_size[0]/my_dpi, img_size[1]/my_dpi), dpi=my_dpi)
    plt.imshow(hue, cmap='Greys_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{name}_fixed.jpg')


# ## Fixed images
# 
# 7 new maps with hillshading removed
# 

# In[ ]:


fixed_images = glob.glob('*fixed.jpg')


# In[ ]:


fixed_images


# In[ ]:


imgs = [Image.open(fixed_image) for fixed_image in fixed_images]


# In[ ]:


imgs[1]


# # Composite map

# In[ ]:


for img in imgs:
    print (img.size, img.filename)


# The images aren't the same size. Dang it. Fortunately the "Potassium", "eThorium" and "eUranium" are the same size.

# In[ ]:


potassium = np.array(imgs[3])
eThorium = np.array(imgs[4])
eUranium = np.array(imgs[1])


# In[ ]:


potassium_gray = np.mean(potassium, axis=2)
eThorium_gray = np.mean(eThorium, axis=2)
eUranium_gray = np.mean(eUranium, axis=2)


# In[ ]:


print('Grayscale image shape:', potassium_gray.shape)
print('Grayscale image shape:', eThorium_gray.shape)
print('Grayscale image shape:', eUranium_gray.shape)


# In[ ]:


# Set the water to np.nan instead of 255
potassium_gray[potassium_gray == np.amax(potassium_gray)] = np.nan
eThorium_gray[eThorium_gray == np.amax(eThorium_gray)] = np.nan
eUranium_gray[eUranium_gray == np.amax(eUranium_gray)] = np.nan


# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(eUranium_gray[::10, ::10], cmap='Greys')
plt.colorbar(shrink=0.5)


# # Corendering K (reds), Th (greens), U (blues)

# In[ ]:


c1, c2, c3 = 255, 255, 255
U_Th_K_stack = np.stack((potassium_gray/c1, eThorium_gray/c2, eUranium_gray/c3), axis=-1)
U_Th_K_stack.shape


# In[ ]:


step = 20  # change this for faster / slower rendering
plt.figure(figsize=(20,20))
plt.imshow(U_Th_K_stack[::step, ::step, :])
plt.savefig('Radiometric_corendering_NS.png')


# In[ ]:


def normU(u):
    """
    A function to scale Uranium map. We don't know what this function should be
    """
    return u


def normTh(th):
    """
    A function to scale thorium.  We don't know what this function should be
    """
    return th


def normK(k):
    """
    A function to scale potassium. We don't know what this function should be
    """
    return k


# In[ ]:


heat_gen1 = heat_equation_no_density(eUranium_gray, eThorium_gray, potassium_gray)


# In[ ]:


heat_gen1[heat_gen1 == np.amax(heat_gen1)] = np.nan


# In[ ]:


plt.figure(figsize=(20,20))
plt.imshow(heat_gen1[::step, ::step], cmap='plasma')
plt.colorbar(shrink=0.5)


# In[ ]:


heat_gen2 = heat_equation(eUranium_gray, eThorium_gray, potassium_gray)


# In[ ]:


heat_gen2[heat_gen2 == np.amax(heat_gen2)] = np.nan


# In[ ]:


plt.figure(figsize=(20,20))
plt.imshow(heat_gen2[::step, ::step], cmap='plasma')
plt.colorbar(shrink=0.5)
plt.savefig('Heat_flow_estimate_NS.png')


# The maps look similar, but are off by about an order of magnitude. Does that help in how they should be scaled?
