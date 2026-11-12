# FreeCT

FreeCT is a collection of tools for reconstructing and working with diagnostic CT projection data.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Packages](#packages)
4. [Licensing](#licensing)
5. [Developer Notes](#notes-from-john)
6. [Development Status](#development-status)
    * [Release Cycle](#release-cycle)
7. [Known Issues](#known-issues-and-limitations)
8. [Contributing](#contributing)
9. [Questions, Comments, Bugs, Feature Requests](#questions-comments-bugs-feature-requests)

## Installation

### Binaries

The easiest way to install FreeCT programs to visit the [Releases](https://gitlab.com/freect/freect/-/releases) page and download the binaries.

Feel free to download and play around.  A good starting place is `./fct-wfbp help`.

**Please see the [known issues](#known-issues-and-limitations) section regarding the "unidentified/unverified developer" warnings on Mac and Windows.**

Although it's a work in progress, please checkout the [quick start](#quick-start) tutorial to run your first reconstruction.

### Source Code

To build from source, you will need to install [Go](https://go.dev/doc/install)

To build *and* install all of the FreeCT programs into your user go/bin folder (on Mac or Linux, this is `~/go/bin`):

```
git clone https://gitlab.com/freect/freect.git
cd freect
go install ./...
```

If you want to build executables individually or without installing, you will need to `cd` into the appropriate directory and build them there.

For example, if we wish to build FreeCT WFBP reconstruction program:

```
cd freect/cmd/fct-wfbp
go build .
./fct-wfbp help
```

All buildable programs for FreeCT can be found in the `cmd` directory.

## Quick Start

This quick start is a work-in-progress.  I will update frequently as I work to shorten it!

#### Step 1: [Install](#installation) a copy of the FreeCT binaries

#### Step 2: Get some DICOM-CT-PD-encoded (a.k.a. "Battelle" format) raw data

You will need access to the "C001" case from the TCIA LDCT and Projection Data dataset to run this tutorial.  Please head to [TCIA's website](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026) and apply for access to the dataset, and download.  Although this is a freely accessible dataset, we are not allowed to redistribute the projection data under the agreement with TCIA. **I will work on securing a sample dataset though so this tutorial does not require this step in the future**

We will assume that the raw data for this case is kept in a directory `/home/user/data/sample-raw-data/`.  You will need to adapt this path for your system depending on where you save the raw data.

Throughout FreeCT, we use the term "battelle" to refer to the raw data encoding found in this TCIA dataset.

#### Step 3: Generate a configuration file for your dataset

Generate the config and dump to standard output (useful for getting a quick impressions of the data as well):
```
./fct-wfbp generate-base-config --input-battelle /home/user/data/sample-raw-data # prints YAML configuration to stdout
```

Redirect the output into a YAML file that we'll provide to the reconstruction:
```
./fct-wfbp generate-base-config --input-battelle=/home/user/data/sample-raw-data > config.yaml # Redirect the output into config.yaml
```

#### Step 4: Make modifications to `config.yaml` 

Open `config.yaml` in a text editor, and change 

```
...
ImageOutput:
    RescaleToHU: false
    AngleOffset: 0
    Binary: ""
    DICOM: ""
    Png: ""
```

to 

```
...
ImageOutput:
    RescaleToHU: true
    AngleOffset: 0
    Binary: ""
    DICOM: "./" # Save dicom to current directory
    Png: ""
```

For your first reconstruction, I'd also recommend changing the start and end positions of the ReconVolume, as well as the field of view:

```
...
ReconVolume:
    StartPosition: -150.0
    EndPosition: -150.0
    FieldOfView: 275.0
    ...
```

This will reconstruct a single slice centered on location `-150.0`, with a FOV (275.0mm) that will allow you to see enough detail to tell if the reconstruction is correct.

Save the file and close.

#### Step 5: Run your reconstruction 

```
./fct-wfbp recon config.yaml
```

#### Step 6: View your images

To view your images, you'll need to manually parse the DICOM data into something like MATLAB's `dicomread` tool.  For some reason, the current DICOM output is not viewable in most DICOM viewers and is an area I'm actively working on.


#### What next?

Playing around with the `ReconVolume`, and `ReconSpecific` parameters is an easy way to explore parameters we commonly adjust in CT reconstruction.  Add more slices, change the slice thickness, tweak kernel settings, etc.  Playing around with `ScanGeometry` and `Preprocessing` blocks will likely result in bad reconstructions, but can be very illuminating when learning about CT!

To see all commands and options for FreeCT WFBP, please use the `--help` option:

```
./fct-wfbp help
./fct-wfbp --help

# or with a specific command

./fct-wfbp recon --help
./fct-wfbp generate-base-config --help
```

## Packages

FreeCT uses Go's module system and users are encouraged to use any of the code in the `pkg` directory in other projects.  We currently have and will continue to expand the following packages:

- `raw` - Raw projection data reading, writing
- `projection` - Structures, methods, and functions for interacting with projection data (once loaded)
- `geometry` - Structures and methods for handling CT scanner geometry
- `image` - Reconstructed image management, manipulation, and reading/writing (including primitive/experimental DICOM support)
- `recon/rebin` - Rebin helical projection data to pseudo-parallel geometry
- `recon/filter` - Filter that projection data!

## Licensing

FreeCT is [licensed for academic and research use](LICENSE.md) under a modified BSD 3-clause license.  The key difference between this licence and the standard BSD 3-clause license is a "citation" requirement for any work that uses or derives from FreeCT.

FreeCT is not currently licensed for industry or for-profit use, however that doesn't mean that it cannot be used for such work.  Please reach out to johnmarianhoffman@gmail.com directly if you would like to use it in industry, and in the meantime please feel free to experiment with the software.

## Notes from John

FreeCT started in 2016 to address the lack of research tools for CT image reconstruction from diagnostic clinical CT scanners.  Other software packages have done, and still do, an excellent job of addressing other CT geometries, namely cone-beam CT, but there's still a lack of tooling for cylindrical, helical CT, which (at least for now) is the dominant geometry in diagnostic CT.

FreeCT was also originally fully written in C++ and CUDA and released as source code that users could compile and run for themselves, however this compilation step, particuarly involving CUDA, proved to be a major obstacle in getting people to use FreeCT.  Furthermore, FreeCT was a complex research tool that even after successful compilation was, at-best, challenging to use, and at worst, moderately inscrutable.  The bigger challenge though was that there was not a viable way for me to continue to support the C++/CUDA version for all of the different platforms folks wanted to use it on.

I have recently chosen to re-implement all of FreeCT using the Go programming language to begin addressing the usability issues.  The new Go implementation has allowed me to implement key functionality that was not possible (primarily due to time) with the C++/CUDA implementation:

- Precompiled binaries that users can download (no user compilation required!)
- Usability of the tools (this is a big one that I'll unpack in more detail elsewhere)
- Take full advantage of modern multi-core CPUs 
- Robust support for multiple raw data formats (namely DICOM-CT-PD, but extension to other formats is planned)
- Supporting tools for interaction and manipulation of CT raw data

And of course high-quality, accurate image reconstruction is the most important.

## Development Status

### Current status: In flux

FreeCT just underwent a massive rewrite to convert the project to the Go programming language.  Although key elements are in place, every time I fix a bug, 1-2 more seem to pop up. It will likely stay that way for a little while as I start to use FreeCT more in my own work.

APIs, interfaces, structures will be kept the same as much as possible between releases, but until we "exercise" FreeCT a little bit more, I cannot promise they will not change.

### GPU Support: Removed (temporarily)

For the time being I have removed GPU-acceleration from FreeCT.  I recognize this could be a deal-breaking change for some folks, however this allows me to dramatically reduce my overhead in terms of project overhead (devops, automation, etc.), and maintenance.

If speed is a big concern for you, take heart: GO is extremely friendly to concurrent programming and I have implemented everything to leverage multicore CPUs where possible.

Although I don't have any official benchmarks yet, my 4-core Macbook Air can backproject a full 512x512 slice in 7-10 seconds.

Finally, I intend to restore GPU functionality when the opportunity or need presents itself (or a talented student/post-doc joins! ;-) ). I have run a positive proof-of-concept integration between CUDA code and Go, however the translation between that and a fully-supported, deployable reconstruction code is non-trivial.

### Release Cycle

FreeCT largely will follow [semantic versioning](https://semver.org/) but will operate on a rolling release schedule.  This means that new versions will come out as often as needed, when they're ready.  Generally speaking though for the foreseeable future, I'm expecting:

- 1-2 patches per month, 
- Monthly minor version bump (as new functionality is added)

Each release is tagged with a release "type" which is one of the following: prerelease, patch, minor, or major.  Users should generally download the latest patch, minor, or major release, and avoid the "prerelease" builds as they are more likely to contain unfixed bugs.

## Known Issues and Limitations

#### Unverified/unidentified developer warnings when using pre-built binaries on Mac and Windows

Plan to resolve: I am currently researching how to secure the required developer certificates and signing procedures so that we can provide valid, signed binaries that the OS will recognize.  This is a high priority issue, however I do not have a timeline for completion.

**Mac workaround**: You will encounter a warning that "fct-wfbp can't be opened because Apple cannot check it for malicious software."  To override this alert, right click on the executable file in the Finder, and select "Open".  You will see a slightly different warning that "macOS cannot verify the developer of 'fct-wfbp.' Are you sure you want to open it?" Click "Open".  You can now run the FreeCT binary normally.

**Windows workaround**: On Windows 10, if you run the executable from the command line, I have not encountered warnings from Windows.  However if you see a "Windows protected your PC" warning, click "More info" and then click the "Run anyway" button that appears.

**Please Note:** Generally speaking, you should not override such alerts from your operating system unless you're 100% sure of what you're doing, and the software that you're doing it for.  If you trust FreeCT, then please feel free to utilize this workaround.  If you do not, or your institutional security policies do not allow for such exceptions, please contact your system administrator for additional guidance.  Other possible workarounds include:[build FreeCT from source](#source-code), on the target system, run FreeCT in a virtual machine, or run FreeCT via a docker container (we will have "official" containers soon).

#### Semi-corrupt DICOM output

FreeCT WFBP can save output images as DICOM, however thus far they are not viewable in most DICOM viewers.  I can read them without issue into MATLAB using `dicominfo` and `dicomread`.  Additionally, there are certain feature of DICOM that FreeCT will not be able to leverage on any foreseeable timeline (e.g., patient orientation).

I will continue to explore what could be causing the viewer issues, but for our own work, I will likely prioritize adding support for other file formats (such as [NIfTI](https://nifti.nimh.nih.gov/) that are commonly used but somewhat simpler and more efficient for research under the hood.  If DICOM is a very high priority for you, please leave that feedback for me on open issue #24, or create a new issue.

#### No support for flying focal spots

Only the "No FFS" case is currently supported.  Next release cycle, I will add support for the "Z FFS" case, which occurs in the TCIA LDCT dataset.  From there, we will investigate the addition of other, slightly less common configurations such as in-plane (a.k.a. Phi FFS), Z+Phi, and diagonal flying focal spot modes.  Order of inclusion into FreeCT will likely be based on demand, so if you have specific requests, please feel free to file an issue, or leave a comment/reaction on an existing flying focal spot related issue.

#### Only partial support for TCIA "LDCT and Projection Data" dataset

Our intent is to have 100% support for the [TCIA LDCT dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026), meaning you can use FreeCT to generate custom reconstructions from any of the projection data found there.  This is still a work in progress.  We should have good support for any of the data from GE scanners (although some manual tweaking may be needed for some cases), and I am actively working on adding support for the Siemens scans (namely adding support for the Z flying focal spot.)

## Contributing

I'd love to have folks contribute to FreeCT, but please keep in mind that I have limited time to review merge requests.

A few things you can do that will help get your MR in:

- Add new code rather than modify existing code
- Keep it short, focused, and narrow scope
- Provide tests for your code (with Go [there's no excuse to not write a test](https://pkg.go.dev/testing))
- Well-written, well-documented code
- A well-written, descriptive MR that address "what," "why," and "how" of your MR

As with the next section, thank you in advance for your patience.

Areas I could use help with:
- Dockerization of FreeCT binaries
- Unit testing current code
- Integration testing reconstruction code
- Automating regression testing reconstruction code
- GPU acceleration

## Questions, Comments, Bugs, Feature Requests

Please provide any such feedback by [filing an issue](https://gitlab.com/freect/freect/-/issues).

Please also understand that primary development efforts of will generally reflect the greatest need in my own academic work, however feedback, feature requests, and bug reports are greatly appreciated.  I will do my best to address them in a timely manner. Thank you in advance for your patience as this is all currently a one-person effort.
