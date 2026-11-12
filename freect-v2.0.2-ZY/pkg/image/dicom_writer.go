package image

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/util"
	dicom "gitlab.com/freect/go-dicom"
	"gitlab.com/freect/go-dicom/pkg/frame"
	"gitlab.com/freect/go-dicom/pkg/tag"
	"gitlab.com/freect/go-dicom/pkg/uid"
	"math"
	"time"
)

type DICOMWriter struct {
	prefix    string
	fileStore ImageStoreWriter
}

var _ ImageWriter = &DICOMWriter{}

// TODO: Move to functional model for options/config handling (like prefix)
func NewDICOMWriter(f ImageStoreWriter, prefix string) *DICOMWriter {
	if prefix == "" {
		prefix = "fct-wfbp"
	}

	return &DICOMWriter{
		prefix:    prefix,
		fileStore: f,
	}
}

func (d *DICOMWriter) String() string {
	return "DICOM image writer for " + d.fileStore.String() + " using prefix \"" + d.prefix + "\""
}

func (d *DICOMWriter) Write(r ImageReader) error {

	log.Warn("DICOM output is experimental and compatibility with all readers and/or tools is NOT guaranteed.")

	images, err := r.Read()
	if err != nil {
		return err
	}

	if !images[0].RescaledToHU {
		log.Warn("Images NOT rescaled to HU. DICOM output will likely result in corrupt images.")
	}

	for i, img := range images {

		dcm_data, err := ImageToDataset(img, i)
		if err != nil {
			return err
		}

		filename := fmt.Sprintf(d.prefix+"-%03d-%.02f-%.02f"+".dcm", i, img.SliceThickness, img.Z)
		wr, err := d.fileStore.Create(filename)
		if err != nil {
			return err
		}
		
		util.LogUpdate(i+1, len(images), len(images), "saving image " + filename)
		
		opts := []dicom.WriteOption{}
		err = dicom.Write(wr, dcm_data, opts...)
		if err != nil {
			return err
		}

	}

	return nil
}

func ImageToDataset(r *ReconstructedImage, instance_number int) (dicom.Dataset, error) {
	SOPClassUID, _ := dicom.NewElement(tag.MediaStorageSOPClassUID, []string{"1.2.840.10008.5.1.4.1.1.2"})
	SOPInstanceUID, _ := dicom.NewElement(tag.MediaStorageSOPInstanceUID, []string{"1.2.3.4.5.6.7"})
	TransferSyntaxUID, _ := dicom.NewElement(tag.TransferSyntaxUID, []string{uid.ExplicitVRLittleEndian})

	dcm_data := dicom.Dataset{
		Elements: []*dicom.Element{
			SOPClassUID,
			SOPInstanceUID,
			TransferSyntaxUID,
		},
	}

	series_time := time.Now()

	//// Metadata
	// Modality
	e, err := dicom.NewElement(tag.Modality, []string{"CT"})
	if err != nil {
		return dcm_data, fmt.Errorf("Set Modality failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// SeriesDate
	e, err = dicom.NewElement(tag.SeriesDate, []string{DateAsString(series_time)})
	if err != nil {
		return dcm_data, fmt.Errorf("Set SeriesDate failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// SliceThickness
	e, err = dicom.NewElement(tag.SliceThickness, []string{fmt.Sprintf("%f", r.SliceThickness)})
	if err != nil {
		return dcm_data, fmt.Errorf("Set SliceThickness failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// SoftwareVersion
	e, err = dicom.NewElement(tag.SoftwareVersion, []string{util.Version()})
	if err != nil {
		return dcm_data, fmt.Errorf("Set SoftwareVersion failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// ReconstructionDiameter
	e, err = dicom.NewElement(tag.ReconstructionDiameter, []string{fmt.Sprintf("%f", r.FieldOfView)})
	if err != nil {
		return dcm_data, fmt.Errorf("Set ReconstructionDiameter failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// ConvolutionKernel
	e, err = dicom.NewElement(tag.ConvolutionKernel, []string{"TODO"})
	if err != nil {
		return dcm_data, fmt.Errorf("Set ConvolutionKernel failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// InstanceNumber
	e, err = dicom.NewElement(tag.InstanceNumber, []string{fmt.Sprintf("%d", instance_number)})
	if err != nil {
		return dcm_data, fmt.Errorf("Set InstanceNumber failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// SliceLocation
	e, err = dicom.NewElement(tag.SliceLocation, []string{fmt.Sprintf("%f", r.Z)})
	if err != nil {
		return dcm_data, fmt.Errorf("Set SliceLocation failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// Rows
	e, err = dicom.NewElement(tag.Rows, []int{r.Ny})
	if err != nil {
		return dcm_data, fmt.Errorf("Set Rows failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// Columns
	e, err = dicom.NewElement(tag.Columns, []int{r.Nx})
	if err != nil {
		return dcm_data, fmt.Errorf("Set Columns failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	//// Image data
	// Convert our float32 image data into int
	rescale_intercept := -1024.0
	rescale_slope := 1.0
	pixel_data := make([]([]int), r.Nx*r.Ny)
	for i, v := range r.Data {
		pixel_data[i] = []int{int(math.Max((float64(v)-rescale_intercept)/rescale_slope, 0.0))}
	}

	// RescaleSlope
	e, err = dicom.NewElement(tag.RescaleSlope, []string{fmt.Sprintf("%f", rescale_slope)})
	if err != nil {
		return dcm_data, fmt.Errorf("Set RescaleSlope failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// RescaleIntercept
	e, err = dicom.NewElement(tag.RescaleIntercept, []string{fmt.Sprintf("%f", rescale_intercept)})
	if err != nil {
		return dcm_data, fmt.Errorf("Set RescaleIntercept failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// SamplesPerPixel
	e, err = dicom.NewElement(tag.SamplesPerPixel, []int{1})
	if err != nil {
		return dcm_data, fmt.Errorf("Set SamplesPerPixel failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// BitsStored & BitsAllocated
	e, err = dicom.NewElement(tag.BitsStored, []int{16})
	if err != nil {
		return dcm_data, fmt.Errorf("Set BitsStored failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	e, err = dicom.NewElement(tag.BitsAllocated, []int{16})
	if err != nil {
		return dcm_data, fmt.Errorf("Set BitsAllocated failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	// Set the pixel data
	e, err = dicom.NewElement(tag.PixelData, dicom.PixelDataInfo{
		IsEncapsulated: false,
		Frames: []frame.Frame{
			{
				Encapsulated: false,
				NativeData: frame.NativeFrame{
					BitsPerSample: 16,
					Rows:          r.Ny,
					Cols:          r.Nx,
					Data:          pixel_data,
				},
			},
		},
	})
	if err != nil {
		return dcm_data, fmt.Errorf("Set PixelData failed: %s", err)
	}
	dcm_data.Elements = append(dcm_data.Elements, e)

	return dcm_data, nil
}

func DateAsString(t time.Time) string {
	return fmt.Sprintf("%04d%02d%02d", t.Year(), int(t.Month()), t.Day())
}
