from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import csv
from source.wsi import WholeSlideImage
import multiprocessing as mp
import hydra
import patchify
import time
from omegaconf import DictConfig, OmegaConf


def single_hdf5_mp(coord, img, attributes, output_path, cfg_region):
    wsi_name=attributes['wsi_name']
    out_path=f'{output_path}/{wsi_name}_{str(coord[0])}_{str(coord[1])}.h5'
    attributes['save_path']=out_path
    #lets make the image region into patches
    patch_dim = (cfg_region.patch_size, cfg_region.patch_size, img.shape[2])
    n_patches_dim = img.shape[0] // cfg_region.patch_size 
    patch= patchify.patchify(img, patch_dim, step=cfg_region.patch_size).reshape(n_patches_dim**2,cfg_region.patch_size,cfg_region.patch_size,3) 
    asset_dict={'coords':coord,'imgs':patch}
    attr_dict={'coords':attributes,'imgs':attributes}
    attr_dict['imgs']['subpatch_size']=cfg_region.patch_size
    result=save_hdf5(out_path, asset_dict, attr_dict, mode="w")
    return result


def save_hdf5(output_path, asset_dict, attr_dict=None, mode="a"):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0] :] = val
    file.close()
    return output_path


def read_region_wrapper(args):
    coord, patch_level, patch_size, wsi_path, attributes, out_path, cfg_region = args
    actual_wsi = WholeSlideImage(wsi_path)
    img = actual_wsi.wsi.read_region(coord, patch_level, patch_size).convert("RGB")
    img = np.array(img)
    result=single_hdf5_mp(np.array(coord), img, attributes, out_path, cfg_region)
    return result


def patch_hd5(coord_dset, wsi_path, attributes,  out_path, cfg_region):
        num_workers = mp.cpu_count()
        if num_workers > 4: 
            num_workers = 4
        patch_level=attributes['patch_level']
        patch_size=attributes['patch_size']
        with mp.Pool(num_workers) as pool:
            iterable = [(tuple(coord), patch_level, (patch_size,patch_size), wsi_path, attributes, out_path, cfg_region) for coord in coord_dset]
            results = pool.map(read_region_wrapper, iterable)
        return results


@hydra.main(
    version_base="1.2.0", config_path="../config/hdf5", config_name="cancer_lgd_config"
)
def main(cfg:DictConfig):
    # read the list of hdf5 files from the csv
    with open(Path(cfg.hdf5_files_csv), 'r') as f:
        reader = csv.reader(f)
        h5_files = [row[0] for row in reader]
        h5_files.remove('h5_files')
    h5_files=[Path(x) for x in h5_files]
    wsi_path = list(Path(cfg.wsi_files_dir).rglob("*.tif"))
    tissue_class =f"{cfg.tissue}_{('_').join(Path(cfg.hdf5_files_csv).stem.split('_')[:-2])}"
    #Check if the csv with processed wsi exists
    processed_list=Path(cfg.processed_wsi_csv)
    if not processed_list.is_file():
        with open(processed_list, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['WSI_name','Label','Number_of_regions', 'Number_of_patches'])
    time_all_class=0
    print(f'Processing {len(h5_files)} WSIs.')
    for index,h5_file in enumerate(h5_files):
        output_path = Path(cfg.output_dir) /tissue_class
        #check if the wsi has been processed
        with open(processed_list, 'r') as f:
            reader = csv.reader(f)
            processed = [row[0] for row in reader]
            processed.remove('WSI_name')
        if h5_file.stem in processed:
            print(f"({index+1}) {h5_file.stem} already processed")
            continue
        #lets time the processing
        start_time = time.time()
        with h5py.File(h5_file, "r") as f:
            dset=f['coords']
            attr_wsi_dict=dict(dset.attrs.items())
            wsi_name=attr_wsi_dict['wsi_name']
            wsi_file=[x for x in wsi_path if wsi_name in x.stem][0]
            output_path = Path(output_path / wsi_name)
            output_path.mkdir(parents=True, exist_ok=True)
            #print(f'({index+1}){wsi_name} has {len(dset)} regions of {attr_wsi_dict["patch_size"]}x{attr_wsi_dict["patch_size"]}')
            results=patch_hd5(dset, wsi_file, attr_wsi_dict, output_path, cfg.region_2_patches)
        total_time = time.time() - start_time
        time_all_class+=total_time
        #add the processed wsi to the csv file with the processed wsi
        with open(processed_list, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([wsi_name, tissue_class, len(results), len(results)*256])
        print(f"({index+1}) {wsi_name}'s {len(results)} regions of {attr_wsi_dict['patch_size']}x{attr_wsi_dict['patch_size']} have been saved in {output_path}")
        print('Time taken: {:.2f} seconds'.format(total_time))
    time_all_class = time.strftime("%H:%M:%S", time.gmtime(time_all_class))
    print(f'Total time taken for {tissue_class} is {time_all_class} (hh:mm:ss)')

if __name__ == "__main__":
    #   python3 HDF5_creation/hdf5_creation.py --config-name 'cancer_lgd_config'
    main()