# Indian-RPNet

The project aims to document the adaptation of [RPNet architecture](https://github.com/detectRecog/CCPD) for Indian vehicle number-plates. The repository also provides a comprehensive Indian ANPR dataset which we have curated while working on this project.

## Description
The project proposes and demonstrates customizations over the RPNet architecture which was originally developed for the CCPD dataset. Before customizing the architecture, we experimented with running [the pre-trained model](https://github.com/detectRecog/CCPD#for-convinence-we-provide-a-trained-wr2-model-and-a-trained-rpnet-model-you-can-download-them-from-google-drive-or-baiduyun) over Indian dataset without much success. We also trained RPNet on CCPD and fine-tuned the model on Indian dataset, which again did not provide promising results. We will be documenting such observations and other related insights in a dedicated blog post soon.

## Getting Started
### Directory structure
* `Code_Chinese`: A copy RPNet and wR2 implementation, for more details, refer the parent repo: https://github.com/detectRecog/CCPD
* `Code_Indian` : Customizations to the RPNet and wR2 implementation for making them work for Indian vehicle number-plates(10-digits only).

### Dependencies

* Python packages:
   * opencv-python==4.2.0.32
   * scikit-build
   * cmake
   * future
   * builtins 
   * imutils
* OS Version: Windows 10

### Installing

* Code:
  * To run the CCPD_Indian version with annotations in the form of .XML files, download the CCPD_Indian code.
* Indian ANPR data:
  * https://www.amazon.com/clouddrive/share/ca4PxbP39VM3cINlV4fv4FVHzbvDOXYARspn3065DFq/folder/Co08Q4pcRI2KV_8xfxvi2w

### Executing program

* Running CCPD_Chinese:

  * train_wR2
  ```
  python train_wR2.py -i <training images folder> -n <number of epochs> -b <batch size> -w <wR2.out>
  ```
  
  * train_rpnet:   
  ```
  python train_rpnet.py -i <training images folder path> -n <number of epochs> -b <batch size> -w <fh02.out> -t <test images folder path> -p <path of wR2.pth> -m <folder in which       model is to be stored>
  ```

## Help
Please contact us via [Github issues](https://github.com/NadimintiSaiSirisha/ANPR/issues) or email at pranav.kant.gaur@gmail.com

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [CCPD](https://github.com/detectRecog/CCPD)

