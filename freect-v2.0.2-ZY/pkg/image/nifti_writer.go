package image

import (
	"errors"
	nii "gitlab.com/freect/freect/pkg/image/nifti"
	"io"
	//"encoding/binary"
	//"bytes"
)

var _ ImageWriter = &Nifti1Writer{}

type Nifti1Writer struct {
	output io.Writer
}

func NewNifti1Writer(output io.Writer) *Nifti1Writer {
	return &Nifti1Writer{
		output: output,
	}
}

func (n *Nifti1Writer) String() string {
	return "NIfTI1 image writer"
}

func (n *Nifti1Writer) Write(r ImageReader) error {

	if n.output == nil {
		return errors.New("write failed: output io.Writer is nil")
	}

	images, err := r.Read()
	if err != nil {
		return err
	}

	// NOTE: Due to the variation in support for scl_slope and scl_inter fields, I'm temporarily
	//       disabling the rescale and just using floating point values.  I'd very much like to add
	//       this back in, however key and common image analysis tools (ITKSNAP, ImageJ, 3DSlicer) all
	//       handle these fields differently and inconsistently.  Given that this whole rewrite is
	//       focused on usability, to me the correct support in each of these tools is more important
	//       at this point than the additional storage that can be saved using the int16 approach.
	
	// Create our underlying nifti image dataset
	//nii_img := nii.NewNifti1Image(images[0].Nx, images[0].Ny, len(images), nii.CodeUnsignedShort)
	////// Stuff images into the nifti structure
	////// TODO: As much as I hate the phrase "code smell," this has a code smell to it.
	////// is there a cleaner way to do this?
	////img_data, slope, intercept := ImageStack(images).CompressToUint16()
	////
	////// Convert the data into bytes
	////b := &bytes.Buffer{}
	////err = binary.Write(b, binary.LittleEndian, img_data)
	////if err != nil {
	////	return err
	////}
	// Set metadata for nifti file based on image stack
	//nii_img.SclSlope = slope
	//nii_img.SclInter = intercept


	nii_img := nii.NewNifti1Image(images[0].Nx, images[0].Ny, len(images), nii.CodeFloat32)
	err = nii_img.SetData(ImageStack(images).Bytes())
	if err != nil {
		return err
	}

	img := images[0]
	dx, dy, dz := float32(img.FieldOfView/float64(img.Nx)), float32(img.FieldOfView/float64(img.Ny)), float32(img.SliceThickness)
	nii_img.SetPixDims(dx, dy, dz)
	
	// TODO: use Xorigin and Yorigin to correctly set offsets in the nifti header

	// Finally, write the image to the supplied io.Writer
	err = nii_img.Write(n.output)
	if err != nil {
		return err
	}
	
	return nil

}
