import cv2
import numpy as np
import glob
import skimage as sk
import skimage.morphology
from skimage.filters import threshold_otsu, threshold_niblack
from skimage import measure
from skimage import io
import holoviews as hv
from holoviews.operation.datashader import datashade, shade
from colorcet import fire, gray
import mahotas as mh
import scipy.ndimage.morphology as morph
import pandas as pd


fire_cmap = np.array(fire)
np.random.shuffle(fire_cmap[1:])
fire_cmap = fire_cmap.tolist()

def rescale(arr):
    im_range = np.max(arr)-np.min(arr) 
    if im_range == 0:
        im_range = 1
    return (arr - np.min(arr))/im_range 

def show_images(img,conn_comps):
    hv_img = plot_raw_image(img)
    hv_conn_comp = plot_conn_comp(conn_comps)
        
    ims = (hv_conn_comp + hv_img).cols(1)

    ims = ims.options({'RGB':dict(width=np.shape(img)[0]//3, height=np.shape(img)[1]//3, fontsize={'title':0, 'xlabel':0, 'ylabel':0, 'ticks':0})})
    return ims.options({'Layout':dict(fontsize={'title':0, 'xlabel':0, 'ylabel':0, 'ticks':0})})

def plot_raw_image(img):
    return shade(hv.Image(rescale(img)),normalization='linear',cmap=gray)   

def plot_conn_comp(conn_comps):
    return shade(hv.Image(rescale(conn_comps)), normalization='linear', cmap=fire_cmap)

def holoplot(img,conn_comps):
    dmap = hv.DynamicMap(show_images(img,conn_comps))
    #dmap = dmap.redim.range(y=(0.5-self.scale_factor_h,0.5), x=(-0.5,-0.5+self.scale_factor_w))
    return dmap
def detect_rough_cells(img,max_perc_contrast = 97):

    def fill_holes(img):
        seed = np.copy(img)
        seed[1:-1, 1:-1] = img.max()
        strel = sk.morphology.square(3, dtype=bool)
        img_filled = sk.morphology.reconstruction(seed, img, selem=strel, method='erosion')
        img_filled = img_filled.astype(bool)
        return img_filled

    max_val = np.percentile(img,max_perc_contrast)
    img_contrast = sk.exposure.rescale_intensity(img, in_range=(0,max_val))

    img_median = mh.median_filter(img_contrast,Bc=np.ones((2,2)))
    img_edges = sk.filters.sobel(img_contrast)

    T_otsu = sk.filters.threshold_otsu(img_edges)
    img_otsu = img_edges > T_otsu
    img_close = morph.binary_closing(img_otsu, structure = np.ones((3,3)),iterations=6)
    img_filled = fill_holes(img_close)
    img_open = morph.binary_opening(img_filled, structure = np.ones((3,3)),iterations=2)
    cell_masks = morph.binary_dilation(img_open,structure = np.ones((9,1)))
    return cell_masks

def get_cell_cutbox(cell_masks, flip_cells = False, cut_from_bottom = 0, above_cell_pad = 0):#90,30
    cell_ccomp = sk.measure.label(cell_masks)
    reg_props = sk.measure.regionprops(cell_ccomp)
    # make sure that in case there is more than one region, we grab the largest region
    rp_area = [r.area for r in reg_props]
    prop = reg_props[np.argmax(rp_area)]
    cutbox = np.zeros(cell_masks.shape,cell_masks.dtype)
    if flip_cells:
        cutbox[prop.bbox[0]+cut_from_bottom:prop.bbox[2]+above_cell_pad,:] = 1
    else:
        cutbox[prop.bbox[0]-above_cell_pad:prop.bbox[2]-cut_from_bottom,:] = 1
    return cutbox

def detect_cells(img, flip_cells = False, max_perc_contrast = 97, cut_from_bottom = 0, above_cell_pad = 0):#90,30
    cell_masks = detect_rough_cells(img,max_perc_contrast = max_perc_contrast)
    cutbox = get_cell_cutbox(cell_masks, flip_cells = flip_cells, cut_from_bottom = cut_from_bottom, above_cell_pad = above_cell_pad)
    
    cell_masks = cell_masks * cutbox
    cell_masks = morph.binary_closing(cell_masks,structure=np.ones((15,1)),iterations=6)
    cell_masks = morph.binary_erosion(cell_masks,structure=np.ones((3,3)),iterations=6)
    return cell_masks
    
    
def extract_connected_components_phase(img, cell_masks = [], flip_cells = False, 
                                       cut_from_bottom = 0, above_cell_pad = 0,
                                       init_smooth_sigma = 3, init_niblack_window_size = 13,
                                       init_niblack_k = -0.35, maxima_smooth_sigma = 2,
                                       maxima_niblack_window_size = 13, maxima_niblack_k = -0.45,
                                       min_cell_size = 30, max_perc_contrast = 97,
                                       return_all = False):

    # makes it directly compatible with fluorescense visualizations (just inverts black and white)
    init_niblack_k = -1*init_niblack_k
    maxima_niblack_k = -1*maxima_niblack_k

    def perform_watershed(threshed, maxima):

        distances = mh.stretch(mh.distance(threshed))
        spots, n_spots = mh.label(maxima,Bc=np.ones((3,3)))
        surface = (distances.max() - distances)

        return sk.morphology.watershed(surface, spots, mask=threshed)


    def findCells(img,mask,sigma,window_size,niblack_k):
        img_smooth = img

        if sigma > 0:    
            img_smooth = sk.filters.gaussian(img,sigma=sigma,preserve_range=True,mode='reflect')

        thresh_niblack = sk.filters.threshold_niblack(img_smooth, window_size = window_size, k= niblack_k)
        threshed = img > thresh_niblack
        threshed = sk.util.invert(threshed)
        threshed = threshed*mask
        return threshed

    def findWatershedMaxima(img,mask):
        maxima = findCells(img,mask,maxima_smooth_sigma,maxima_niblack_window_size,maxima_niblack_k)
#         maxima = mh.erode(maxima,Bc=np.ones((7,5))) #Initial
#         maxima = mh.erode(maxima,Bc=np.ones((5,3)))
        reg_props = sk.measure.regionprops(sk.measure.label(maxima,neighbors=4))
        # make sure that in case there is more than one region, we grab the largest region
        rp_area = [r.area for r in reg_props]
        med_size = np.median(rp_area)
        std_size = np.std(rp_area)
        cutoff_size = int(max(0,med_size/6))
        
        maxima = sk.morphology.remove_small_objects(maxima,min_size=cutoff_size)
        return maxima

    def findWatershedMask(img,mask):
        img_mask = findCells(img,mask,init_smooth_sigma,init_niblack_window_size,init_niblack_k)
        img_mask = mh.dilate(mh.dilate(img_mask),Bc=np.ones((1,3),dtype=np.bool))
        img_mask = sk.morphology.remove_small_objects(img_mask,min_size=4)
        return img_mask
    
    
    if len(cell_masks) == 0:
        cell_masks = detect_cells(img, flip_cells = flip_cells, max_perc_contrast = max_perc_contrast, cut_from_bottom = cut_from_bottom, above_cell_pad = above_cell_pad)
    
    img_median = mh.median_filter(img,Bc=np.ones((3,3)))
    img_mask = findWatershedMask(img_median,cell_masks)
    maxima = findWatershedMaxima(img_median,img_mask)

    conn_comp = perform_watershed(img_mask, maxima)

    # re-label in case regions are split during multiplication
    conn_comp = sk.measure.label(conn_comp,neighbors=4)
    conn_comp = sk.morphology.remove_small_objects(conn_comp,min_size=min_cell_size)

    if return_all:
        return conn_comp, cell_masks, img_mask, maxima
    else:
        return conn_comp

def calc_mean_top_n_percent(prop, n_percent):

    topN = np.percentile(prop.intensity_image[:], 100 - n_percent)
    meanTopN = np.mean(prop.intensity_image[prop.intensity_image > topN])
    return meanTopN

def set_region_properties(prop_dict,prop_values,prop_names,prefix=''):

    if type(prop_values) is np.ndarray:
        assert len(prop_values) == len(prop_names), "prop values and prop names are not the same length"
        for i,pname in enumerate(prop_names):
            prop_dict[prefix + pname] = prop_values[i]
    else:
        for pname in prop_names:
            prop_dict[prefix + pname] = getattr(prop_values,pname)
            
    return prop_dict
        
def set_file_properties(prop_dict, prop, prefix=''):

    file_prop_names = ['filename','img_dir']

    return set_region_properties(prop_dict,prop,file_prop_names,prefix=prefix)

def set_image_dim_properties(prop_dict, img_height=np.nan, img_width=np.nan, prefix=''):

    image_dim_prop_names = ['img_height','img_width']

    return set_region_properties(prop_dict,np.array([img_height,img_width]),image_dim_prop_names,prefix=prefix)


def set_fl_file_properties(prop_dict, props, prefixes=['fl0_','fl1_','fl2_']):

    file_prop_names = ['filename']
    n_fluor_images = len(props)
    
    for i in range(len(prefixes)):    
        if i <= (n_fluor_images - 1):
            fl_file = props[i]
        else:
            fl_file = np.array([''])
        prop_dict = set_region_properties(prop_dict, fl_file, file_prop_names, prefixes[i])
    
    prop_dict = set_region_properties(prop_dict,np.array([n_fluor_images]),['n_fluorescent_images'])
    
    return prop_dict

def set_morphological_properties(prop_dict, prop, prefix='',props_to_grab='all'):

    minimal_prop_names = ['area','bbox','centx','centy','label',
                          'major_axis_length','minor_axis_length']

    supp_prop_names = ['convex_area','eccentricity','equivalent_diameter',
                       'euler_number','extent','filled_area',
                       'orientation','perimeter','solidity']

    if props_to_grab == 'min':
        morphological_prop_names = minimal_prop_names
    elif props_to_grab == 'supp':
        morphological_prop_names = supp_prop_names
    else:
        minimal_prop_names.extend(supp_prop_names)
        morphological_prop_names = minimal_prop_names

    prop.centx = prop.centroid[1]
    prop.centy = prop.centroid[0]

    return set_region_properties(prop_dict,prop,morphological_prop_names,prefix=prefix)

def set_intensity_properties(prop_dict,prop,prefix=''):

    intensity_prop_names = ['intensity_image','max_intensity','mean_intensity',
                        'mean_top_30_percent','min_intensity']
    
    if prop is np.nan:
        prop = np.array([np.nan]*len(intensity_prop_names))
    elif type(prop) is skimage.measure._regionprops._RegionProperties:
        prop.mean_top_30_percent = calc_mean_top_n_percent(prop, 30)
    else:
        print(type(prop))
    
    return set_region_properties(prop_dict, prop, intensity_prop_names, prefix=prefix)

def extract_cells(img, connected_components, fluorescent_files = [],props_to_grab='all'):

    fl_prefixes = []
    for i in range(len(fluorescent_files)):
        fl_prefixes.append('fl'+str(i)+'_')
        
    seg_img = img
    seg_props = measure.regionprops(connected_components,intensity_image=seg_img)

    fl_props = []
    for fl_file in fluorescent_files:
        fl_props.append(measure.regionprops(connected_components,intensity_image= sk.io.imread(fl_file)))

    # intialize property dictionary so that all regions have file info
    prop_dict0 = {}
    prop_dict0 = set_image_dim_properties(prop_dict0, img_height=seg_img.shape[0], img_width = seg_img.shape[1])
    
    for i in range(len(fluorescent_files)):
        prop_dict0 = set_intensity_properties(prop_dict0,np.nan,prefix=fl_prefixes[i])
        
    props = []
    for cell_idx, prop in enumerate(seg_props):
        prop_dict = dict(prop_dict0) # do not change this or it will overwrite everytime
        prop_dict = set_morphological_properties(prop_dict, prop,props_to_grab=props_to_grab)
        prop_dict = set_intensity_properties(prop_dict,prop)
        
        for i in range(len(fluorescent_files)):
            prop_dict = set_intensity_properties(prop_dict,fl_props[i][cell_idx],fl_prefixes[i])
            
        props.append(prop_dict)

    return props
