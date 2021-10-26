# Project Title

The project aims to document the adaptation of [RPNet architecture](https://github.com/detectRecog/CCPD) for Indian schenario. The repository also provides a comprehensive Indian ANPR dataset which we curated while working on this project.

## Description
The project proposes and demonstrates customizations over the RPNet architecture which was originally developed for the CCPD dataset. Before customizing the architecture, we experimented with running [the pre-trained model](https://github.com/detectRecog/CCPD#for-convinence-we-provide-a-trained-wr2-model-and-a-trained-rpnet-model-you-can-download-them-from-google-drive-or-baiduyun) over Indian dataset without much success. We also trained RPNet on CCPD and fine-tuned the model on Indian dataset, which again did not provide promising results. We will also be documenting the such observations and other related insights in a dedicated blog post soon.

## Getting Started

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

* To run the CCPD_Indian version with annotations in the form of .XML files, download the CCPD_Indian code.
* 
* Any modifications needed to be made to files/folders

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


* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
