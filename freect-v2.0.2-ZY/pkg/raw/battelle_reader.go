package raw

import (
	"errors"
	"fmt"
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/projections"
	dicom "gitlab.com/freect/go-dicom"
	"gitlab.com/freect/go-dicom/pkg/tag"
	"strconv"
	//"gopkg.in/yaml.v3"
	"gitlab.com/freect/freect/pkg/util"
)

var _ RawReader = &BattelleReader{}

// BattelleReader reads raw projection data from the "Battelle" format
// specified by Mayo Clinic.  It fulfills the RawReader interface.
type BattelleReader struct {
	//path     string
	lastRead int
	fileList []string
	reader   FileStoreReader
}

// Helper function to reduce boilerplate
func (d *BattelleReader) readDatasetFromStore(curr_file string) (dicom.Dataset, error) {

	file, err := d.reader.Open(curr_file)
	if err != nil {
		return dicom.Dataset{}, err
	}

	info, err := d.reader.Stat(curr_file)
	if err != nil {
		return dicom.Dataset{}, err
	}

	dcm_data, err := dicom.Parse(file, info.Size(), nil)
	if err != nil {
		return dicom.Dataset{}, err
	}

	err = file.Close()
	if err != nil {
		return dicom.Dataset{}, err
	}

	return dcm_data, nil
}

func NewBattelleReader(r FileStoreReader) (*BattelleReader, error) {
	file_list, err := r.ListFiles()
	if err != nil {
		return nil, err
	}

	dr := &BattelleReader{
		lastRead: -1,
		fileList: file_list,
		reader:   r,
	}

	return dr, nil
}

func (d *BattelleReader) String() string {
	return fmt.Sprintf("Battelle format reader, %s", d.reader.String())
}

func (d *BattelleReader) ReadProjection() (*projections.Projection, error) {
	d.lastRead++

	// No more images
	if d.lastRead == len(d.fileList) {
		return nil, nil
	}

	curr_file := d.fileList[d.lastRead]

	dcm_data, err := d.readDatasetFromStore(curr_file)
	if err != nil {
		return nil, err
	}

	p := &projections.Projection{
		Index: d.lastRead,
	}

	// Extract metadata
	e, err := dcm_data.FindElementByTag(tag.NumberofDetectorRows)
	if err != nil {
		return nil, err
	}
	p.NumRows = int(e.Value.GetValue().([]int)[0])

	e, err = dcm_data.FindElementByTag(tag.NumberofDetectorColumns)
	if err != nil {
		return nil, err
	}
	p.NumChannels = int(e.Value.GetValue().([]int)[0])

	e, err = dcm_data.FindElementByTag(tag.DetectorFocalCenterAngularPosition)
	if err != nil {
		return nil, err
	}
	p.TubeAngle = float64(e.Value.GetValue().([]float64)[0])

	e, err = dcm_data.FindElementByTag(tag.DetectorFocalCenterAxialPosition)
	if err != nil {
		return nil, err
	}
	p.TablePosition = float64(e.Value.GetValue().([]float64)[0])

	// Extract central detector (a little more complicated)
	var central_detector []float64 = nil
	if err := readFloatsTag(&dcm_data, tag.DetectorCentralElement, &central_detector); err != nil {
		return nil, err
	}

	if len(central_detector) != 2 {
		err_msg := fmt.Sprintf("expected two coordinates for central detector, but found %d", len(central_detector))
		return nil, errors.New(err_msg)
	}
	p.CentralChannel = (central_detector)[0]
	p.CentralRow = (central_detector)[1]

	// Get the detector size
	if err := readFloatTag(&dcm_data, tag.DetectorElementTransverseSpacing, &p.ChannelWidth); err != nil {
		return nil, err
	}

	if err := readFloatTag(&dcm_data, tag.DetectorElementAxialSpacing, &p.RowWidth); err != nil {
		return nil, err
	}

	// Extract pixel data
	p.Data, err = readPixelData(&dcm_data, p.NumRows, p.NumChannels)
	if err != nil {
		return nil, err
	}

	return p, nil
}

func (d *BattelleReader) ReadGeometry() (*geometry.ScanGeometry, error) {

	s := &geometry.ScanGeometry{}

	// Grab our first file in the file list
	curr_file := d.fileList[0]

	// Load the projection
	file, err := d.reader.Open(curr_file)
	if err != nil {
		return nil, err
	}

	info, err := d.reader.Stat(curr_file)
	if err != nil {
		return nil, err
	}

	dcm_data, err := dicom.Parse(file, info.Size(), nil)
	if err != nil {
		return nil, err
	}

	err = file.Close()
	if err != nil {
		return nil, err
	}

	// Parse the necessary tags
	if err := readStringTag(&dcm_data, tag.Manufacturer, &s.Manufacturer); err != nil {
		return nil, err
	}

	if err := readIntTag(&dcm_data, tag.NumberofDetectorRows, &s.DetectorRows); err != nil {
		return nil, err
	}

	if err := readIntTag(&dcm_data, tag.NumberofDetectorColumns, &s.DetectorChannels); err != nil {
		return nil, err
	}

	if err := readFloatTag(&dcm_data, tag.DetectorElementTransverseSpacing, &s.DetectorTransverseSpacing); err != nil {
		return nil, err
	}

	if err := readFloatTag(&dcm_data, tag.DetectorElementAxialSpacing, &s.DetectorAxialSpacing); err != nil {
		return nil, err
	}

	if err := readStringTag(&dcm_data, tag.DetectorShape, &s.DetectorShape); err != nil {
		return nil, err
	}

	if err := readFloatTag(&dcm_data, tag.DetectorFocalCenterRadialDistance, &s.DistSourceToIsocenter); err != nil {
		return nil, err
	}

	if err := readFloatTag(&dcm_data, tag.ConstantRadialDistance, &s.DistSourceToDetector); err != nil {
		return nil, err
	}

	if err := readIntTag(&dcm_data, tag.NumberofSourceAngularSteps, &s.ProjectionsPerRotation); err != nil {
		return nil, err
	}

	if err := readStringTag(&dcm_data, tag.TypeofProjectionData, &s.ScanType); err != nil {
		return nil, err
	}

	if err := readStringTag(&dcm_data, tag.FlyingFocalSpotMode, &s.FlyingFocalSpotMode); err != nil {
		return nil, err
	}

	if err := readStringTag(&dcm_data, tag.TypeofProjectionGeometry, &s.ProjectionGeometry); err != nil {
		return nil, err
	}

	if err := readFloatTag(&dcm_data, tag.DetectorFocalCenterAxialPosition, &s.StartTablePosition); err != nil {
		return nil, err
	}

	s.AcquisitionFieldOfView, err = readDecimalString(&dcm_data, tag.DataCollectionDiameter)
	if err != nil {
		return nil, err
	}

	// There seems to be some disagreement between the dictionary
	// provided with the LDCT PD data and the publication about what
	// this tag is. We try to read either one but don't fail if we can't find it.
	var HUCalibrationTag = tag.Tag{0x0018, 0x0061}
	s.HUCalibrationFactor, err = readDecimalString(&dcm_data, HUCalibrationTag)
	if err != nil {
		s.HUCalibrationFactor, err = readDecimalString(&dcm_data, tag.WaterAttenuationCoefficient)
		if err != nil {
			log.Warn("Could not find HU Calibration Factor in dataset.  HU rescaling will not be possible.")
		}
	}

	//if err := readFloatTag(&dcm_data, tag.DataCollectionDiameter, &s.AcquisitionFieldOfView); err != nil {
	//	fmt.Println(err)
	//	return nil, err
	//}

	s.CollimatedSliceWidth = s.DistSourceToIsocenter * s.DetectorAxialSpacing / s.DistSourceToDetector

	pitch := 0.0
	if err := readFloatTag(&dcm_data, tag.SpiralPitchFactor, &pitch); err != nil {
		fmt.Println(err)
		return nil, err
	}
	//fmt.Printf("SpiralPitchFactor: %v\n", pitch)
	s.TableFeedPerRotation = pitch * s.CollimatedSliceWidth * float64(s.DetectorRows)

	var central_detector []float64 = nil
	if err := readFloatsTag(&dcm_data, tag.DetectorCentralElement, &central_detector); err != nil {
		return nil, err
	}

	if len(central_detector) != 2 {
		err_msg := fmt.Sprintf("expected two coordinates for central detector, but found %d", len(central_detector))
		return nil, errors.New(err_msg)
	}
	s.DetectorCentralChannel = (central_detector)[0]
	s.DetectorCentralRow = (central_detector)[1]

	s.NumProjectionsTotal = len(d.fileList)

	// Grab our last file in the file list
	last_file := d.fileList[len(d.fileList)-1]

	dcm_data, err = d.readDatasetFromStore(last_file)
	if err != nil {
		return nil, err
	}

	if err := readFloatTag(&dcm_data, tag.DetectorFocalCenterAxialPosition, &s.EndTablePosition); err != nil {
		return nil, err
	}

	return s, nil
}

func (d *BattelleReader) ReadN(N int) ([]*projections.Projection, *geometry.ScanGeometry, error) {

	projs := make([]*projections.Projection, N)
	for i := range projs {
		if i == N {
			break
		}

		util.LogUpdate(i + 1, N, 10, "reading projections")
		
		p, err := d.ReadProjection()
		if err != nil {
			return nil, nil, err
		}
		projs[i] = p
	}

	g, err := d.ReadGeometry()
	if err != nil {
		return nil, nil, err
	}

	g.NumProjectionsTotal = N

	//data, _ := yaml.Marshal(g)
	//fmt.Println(string(data))

	return projs, g, nil

}

func (d *BattelleReader) Read(geom_overrides *geometry.ScanGeometry) ([]*projections.Projection, *geometry.ScanGeometry, error) {
	projs := make([]*projections.Projection, len(d.fileList))
	for i := range projs {
		p, err := d.ReadProjection()
		if err != nil {
			return nil, nil, err
		}

		util.LogUpdate(i + 1, len(d.fileList), 10, "reading projections")
		
		projs[i] = p
	}

	g, err := d.ReadGeometry()
	if err != nil {
		return nil, nil, err
	}

	if geom_overrides == nil {
		return projs, g, nil
	}

	// TODO: Somehow alert the user that overrides are being utilized?
	g = geom_overrides
	for _, p := range projs {
		p.CentralRow = geom_overrides.DetectorCentralRow
		p.CentralChannel = geom_overrides.DetectorCentralChannel
		p.ChannelWidth = geom_overrides.DetectorTransverseSpacing
		p.RowWidth = geom_overrides.DetectorAxialSpacing
	}

	return projs, g, nil
}

func readDecimalString(data *dicom.Dataset, t tag.Tag) (float64, error) {
	e, err := data.FindElementByTag(t)
	if err != nil {
		return 0.0, err
	}

	string_val := e.Value.GetValue().([]string)[0]

	val, err := strconv.ParseFloat(string_val, 64)
	if err != nil {
		return 0.0, err
	}

	return val, nil
}

func readPixelData(dcm_data *dicom.Dataset, detector_rows int, detector_channels int) (final_proj []float32, err error) {
	// Extract rescale values for pixel data
	rescale_intercept, err := readDecimalString(dcm_data, tag.RescaleIntercept)
	if err != nil {
		return nil, err
	}

	rescale_slope, err := readDecimalString(dcm_data, tag.RescaleSlope)
	if err != nil {
		return nil, err
	}

	//log.Info("Rescale slope: ", rescale_slope)
	//log.Info("Rescale intercept: ", rescale_intercept)

	// Extract pixel data
	e, err := dcm_data.FindElementByTag(tag.PixelData)
	imageInfo := e.Value.GetValue().(dicom.PixelDataInfo)

	for _, f := range imageInfo.Frames {
		nf, err := f.GetNativeFrame()
		if err != nil {
			return nil, err
		}

		final_proj = make([]float32, nf.Rows*nf.Cols)

		if len(nf.Data[0]) > 1 {
			return nil, errors.New("multiple channels found in projection data (should be grayscale)")
		}

		// Determine if pixel data was incorrectly stored transposed
		if nf.Rows == detector_channels && nf.Cols == detector_rows {
			for i := 0; i < detector_rows; i++ {
				for j := 0; j < detector_channels; j++ {
					dest_idx := j + i*detector_channels
					src_idx := i + j*detector_rows
					final_proj[dest_idx] = float32(rescale_intercept) + float32(rescale_slope)*float32(nf.Data[src_idx][0])
				}
			}
		} else {
			for i := range final_proj {
				final_proj[i] = float32(rescale_intercept) + float32(rescale_slope)*float32(nf.Data[i][0])
			}
		}

	}

	return final_proj, nil
}

func readStringTag(d *dicom.Dataset, t tag.Tag, s *string) error {

	e, err := d.FindElementByTag(t)
	if err != nil {
		return err
	}

	strs, ok := e.Value.GetValue().([]string)
	if !ok {
		err_msg := fmt.Sprintf("string assertion failed for tag: %s", t.String())
		return errors.New(err_msg)
	}

	if len(strs) < 1 {
		err_msg := fmt.Sprintf("no strings returned for tag: %s", t.String())
		return errors.New(err_msg)
	}

	*s = strs[0]

	return nil
}

func readIntTag(d *dicom.Dataset, t tag.Tag, i *int) error {

	e, err := d.FindElementByTag(t)
	if err != nil {
		return err
	}

	ints, ok := e.Value.GetValue().([]int)
	if !ok {
		err_msg := fmt.Sprintf("int assertion failed for tag: %s", t.String())
		return errors.New(err_msg)
	}

	if len(ints) < 1 {
		err_msg := fmt.Sprintf("no ints returned for tag: %s", t.String())
		return errors.New(err_msg)
	}

	*i = ints[0]

	return nil
}

func readFloatTag(d *dicom.Dataset, t tag.Tag, f *float64) error {

	e, err := d.FindElementByTag(t)
	if err != nil {
		return err
	}

	floats, ok := e.Value.GetValue().([]float64)
	if !ok {
		err_msg := fmt.Sprintf("float64 assertion failed for tag: %s", t.String())
		return errors.New(err_msg)
	}

	if len(floats) < 1 {
		err_msg := fmt.Sprintf("no float64s returned for tag: %s", t.String())
		return errors.New(err_msg)
	}

	*f = floats[0]

	return nil
}

func readFloatsTag(d *dicom.Dataset, t tag.Tag, f *[]float64) error {

	e, err := d.FindElementByTag(t)
	if err != nil {
		return err
	}

	floats, ok := e.Value.GetValue().([]float64)
	if !ok {
		err_msg := fmt.Sprintf("float64 assertion failed for tag: %s", t.String())
		return errors.New(err_msg)
	}

	if len(floats) == 0 {
		err_msg := fmt.Sprintf("no float64s returned for tag: %s", t.String())
		return errors.New(err_msg)
	}

	*f = floats

	return nil

}
