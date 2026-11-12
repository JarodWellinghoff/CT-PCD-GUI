package nifti

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"io"
)

// TODO move all of the casting to these antiquated types into the
// nifti methods so that we don't litter the FreeCT code with a bunch
// of casting. I.e. Functions should accept and return float64s and ints
// and handle any casting internally to get them into types the header needs.

// Nifti1 header implements the concrete header structure for nifti
// files.  The resulting structure serializes into and out of
// io.Writer/Reader well using binary.Write.
type Nifti1Header struct {
	SizeofHdr     int32      /*!< MUST be 348           */
	DataType      [10]byte   /*!< ++UNUSED++            */
	DbName        [18]byte   /*!< ++UNUSED++            */
	Extents       int32      /*!< ++UNUSED++            */
	SessionError  int16      /*!< ++UNUSED++            */
	Regular       byte       /*!< ++UNUSED++            */
	DimInfo       byte       /*!< MRI slice ordering.   */
	Dim           [8]int16   /*!< Data array dimensions.*/
	IntentP1      float32    /*!< 1st intent parameter. */
	IntentP2      float32    /*!< 2nd intent parameter. */
	IntentP3      float32    /*!< 3rd intent parameter. */
	IntentCode    int16      /*!< NIFTI_INTENT_* code.  */
	Datatype      int16      /*!< Defines data type!    */
	Bitpix        int16      /*!< Number bits/voxel.    */
	SliceStart    int16      /*!< First slice index.    */
	Pixdim        [8]float32 /*!< Grid spacings.        */
	VoxOffset     float32    /*!< Offset into .nii file */
	SclSlope      float32    /*!< Data scaling: slope.  */
	SclInter      float32    /*!< Data scaling: offset. */
	SliceEnd      int16      /*!< Last slice index.     */
	SliceCode     byte       /*!< Slice timing order.   */
	XyztUnits     byte       /*!< Units of pixdim[1..4] */
	CalMax        float32    /*!< Max display intensity */
	CalMin        float32    /*!< Min display intensity */
	SliceDuration float32    /*!< Time for 1 slice.     */
	Toffset       float32    /*!< Time axis shift.      */
	Glmax         int32      /*!< ++UNUSED++            */
	Glmin         int32      /*!< ++UNUSED++            */
	Descrip       [80]byte   /*!< any text you like.    */
	AuxFile       [24]byte   /*!< auxiliary filename.   */
	QformCode     int16      /*!< NIFTI_XFORM_* code.   */
	SformCode     int16      /*!< NIFTI_XFORM_* code.   */
	QuaternB      float32    /*!< Quaternion b param.   */
	QuaternC      float32    /*!< Quaternion c param.   */
	QuaternD      float32    /*!< Quaternion d param.   */
	QoffsetX      float32    /*!< Quaternion x shift.   */
	QoffsetY      float32    /*!< Quaternion y shift.   */
	QoffsetZ      float32    /*!< Quaternion z shift.   */
	SrowX         [4]float32 /*!< 1st row affine transform.   */
	SrowY         [4]float32 /*!< 2nd row affine transform.   */
	SrowZ         [4]float32 /*!< 3rd row affine transform.   */
	IntentName    [16]byte   /*!< 'name' or meaning of data.  */
	Magic         [4]byte    /*!< MUST be "ni1\0" or "n+1\0". */

} /**** 348 bytes total ****/

// We implement a few methods for fields that have non-obvious access
// patterns.  Generally though, most fields should just be set
// directly.
func (nh *Nifti1Header) SetDims(nx, ny, nz int) {
	nh.Dim[0] = int16(3)
	nh.Dim[1] = int16(nx)
	nh.Dim[2] = int16(ny)
	nh.Dim[3] = int16(nz)

	//nh.Dim[4] = 1.0
	//nh.Dim[5] = 1.0
	//nh.Dim[6] = 1.0
	//nh.Dim[7] = 1.0
}

func (nh *Nifti1Header) NDim() int16 {
	return nh.Dim[0]
}

func (nh *Nifti1Header) Nx() int16 {
	return nh.Dim[1]
}

func (nh *Nifti1Header) Ny() int16 {
	return nh.Dim[2]
}

func (nh *Nifti1Header) Nz() int16 {
	return nh.Dim[3]
}

func (nh *Nifti1Header) SetPixDims(dx, dy, dz float32) {
	nh.Pixdim[0] = 0.0
	nh.Pixdim[1] = dx
	nh.Pixdim[2] = dy
	nh.Pixdim[3] = dz

	//nh.Pixdim[4] = 1.0
	//nh.Pixdim[5] = 1.0
	//nh.Pixdim[6] = 1.0
	//nh.Pixdim[7] = 1.0
}

func (nh *Nifti1Header) SetDescription(msg string) error {

	if len(msg) > 80 {
		return errors.New("description must be <= 80 characters")
	}

	copy(nh.Descrip[:], []byte(msg))

	return nil
}

func (nh *Nifti1Header) SetMeasurementUnit(xyz_dtype uint8, t_dtype uint8) {
	nh.XyztUnits = byte(xyz_dtype | t_dtype) // This is GUESS. I have no idea if this is the intended approach.
}

func (nh *Nifti1Header) SetDatatype(dtype int16) {

	nh.Datatype = dtype

	switch nh.Datatype {
	case CodeUnknown:
		nh.Bitpix = BitpixUnknown
	case CodeBool:
		nh.Bitpix = BitpixBool
	case CodeUnsignedChar:
		nh.Bitpix = BitpixUnsignedChar
	case CodeSignedShort:
		nh.Bitpix = BitpixSignedShort
	case CodeSignedInt:
		nh.Bitpix = BitpixSignedInt
	case CodeFloat32:
		nh.Bitpix = BitpixFloat32
	case CodeComplex64:
		nh.Bitpix = BitpixComplex64
	case CodeDouble64:
		nh.Bitpix = BitpixDouble64
	case CodeRGB:
		nh.Bitpix = BitpixRGB
	case CodeAll:
		nh.Bitpix = BitpixAll
	case CodeSignedChar:
		nh.Bitpix = BitpixSignedChar
	case CodeUnsignedShort:
		nh.Bitpix = BitpixUnsignedShort
	case CodeUnsignedint:
		nh.Bitpix = BitpixUnsignedInt
	case CodeLongLong:
		nh.Bitpix = BitpixLongLong
	case CodeUnsignedLongLong:
		nh.Bitpix = BitpixUnsignedLongLong
	case CodeLongDouble:
		nh.Bitpix = BitpixLongDouble
	case CodeDoublePair:
		nh.Bitpix = BitpixDoublePair
	case CodeLongDoublePair:
		nh.Bitpix = BitpixLongDoublePair
	case CodeRGBA:
		nh.Bitpix = BitpixLongDoublePair
	}

}

type Nifti1Image struct {
	Nifti1Header
	pad  [4]byte
	data []byte
}

// SetData sets the data block of the nifti image.  This puts the onus
// on the user to serialize the data correctly first.
func (ni *Nifti1Image) SetData(data []byte) error {
	if len(data) != len(ni.data) {
		return errors.New("invalid data. len(data) != expected length for nifti image")
	}
	ni.data = data
	return nil
}

// SetSliceData sets the data block of the nifti image for the slice
// specified.  This puts the onus on the user to serialize the slice
// data correctly first.
func (ni *Nifti1Image) SetSliceData(slice_idx int, data []byte) error {

	slice_size := int(ni.Nx()) * int(ni.Ny()) * int(ni.Bitpix) / 8

	if len(data) != slice_size {
		return errors.New("invalid data. len(data) != expected length for nifti slice")
	}

	offset := slice_idx * slice_size
	copy(ni.data[offset:(offset+slice_size)], data)
	return nil
}

// Write serializes the nifti1 header and data into the supplied
// io.Writer Output is compressed with gzip using the
// gzip.BestCompression setting.
func (ni *Nifti1Image) Write(output io.Writer) error {

	// Have to use the temp buffer here since binary can't write the
	// whole image struct in one go and gzip needs to operate on the
	// whole file

	// Binary serializes our data into our temporary buffer
	tmp_buff := &bytes.Buffer{}

	err := binary.Write(tmp_buff, binary.LittleEndian, ni.Nifti1Header)
	if err != nil {
		return err
	}

	err = binary.Write(tmp_buff, binary.LittleEndian, ni.pad)
	if err != nil {
		return err
	}

	err = binary.Write(tmp_buff, binary.LittleEndian, ni.data)
	if err != nil {
		return err
	}

	// GZip writer writes into our file
	gzip_wr, err := gzip.NewWriterLevel(output, gzip.BestCompression)
	if err != nil {
		return err
	}

	// Grab the data out of our temporary buffer and save it to our
	// file Note the defer close here is not just good practice,
	// gzipped files will be corrupt (not fully flushed) without that
	// line of code.
	_, err = gzip_wr.Write(tmp_buff.Bytes())
	defer gzip_wr.Close()
	if err != nil {
		return err
	}

	return nil
}

// NewNifti1Image creates a new nx*ny*nz nifti v1 image structure with
// underlying datatype.  The int16 for datatype should be one of the
// constants supplied in nifti_consts.go.
func NewNifti1Image(nx, ny, nz int, datatype int16) *Nifti1Image {

	ni_img := &Nifti1Image{}
	ni_img.SizeofHdr = 348
	ni_img.VoxOffset = float32(352.0)
	ni_img.SetDims(nx, ny, nz)
	ni_img.SetPixDims(1.0, 1.0, 1.0)
	ni_img.SetDatatype(datatype)
	ni_img.SetDescription("Image created with FreeCT!")
	ni_img.SetMeasurementUnit(Millimeter, Seconds)
	ni_img.Magic = [4]byte{0x6e, 0x2b, 0x32, 0x00} // "n+1" according to the nifti spec
	ni_img.data = make([]byte, nx*ny*nz*(int(ni_img.Bitpix)/8))

	ni_img.SclSlope = 1.0
	ni_img.SclInter = 0.0
	//ni_img.CalMax = float32(math.Pow(2.0, 16.0)) //255.0
	//ni_img.CalMax = 7000.0
	//ni_img.CalMin = 3000.0

	ni_img.QformCode = 1
	ni_img.SformCode = 0
	//
	//ni_img.QuaternC = 1.0
	//
	//ni_img.SrowX = [4]float32{1.0, 0.0, 0.0, 0.0}
	//ni_img.SrowY = [4]float32{0.0, 1.0, 0.0, 0.0}
	//ni_img.SrowZ = [4]float32{0.0, 0.0, 1.0, 0.0}

	return ni_img
}
