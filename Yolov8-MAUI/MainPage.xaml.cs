
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using Yolov8_MAUI.Models;
using IImage = Microsoft.Maui.Graphics.IImage;
using SixLabors.ImageSharp.Formats;
using Microsoft.Maui.Graphics.Platform;
using System.Linq;

namespace Yolov8_MAUI;

public partial class MainPage : ContentPage
{
    private List<string> _classNames { get; set; }

    private InferenceSession _session { get; set; }

    private float _threshold = 0.5f;
  
    public MainPage()
	{
		InitializeComponent();
        _classNames = LoadLabels();
        _session = LoadModel();
    }

    private InferenceSession LoadModel() 
    {
        using var modelStream = FileSystem.OpenAppPackageFileAsync("yolov8n.onnx").Result;

        using var modelMemoryStream = new MemoryStream();
        modelStream.CopyTo(modelMemoryStream);

        var _model = modelMemoryStream.ToArray();
        InferenceSession inferenceSession = new InferenceSession(model:_model);

        return inferenceSession;
    }

    private List<string> LoadLabels()
    {
        // Loading the labels
        using var stream = FileSystem.OpenAppPackageFileAsync("labels.txt").Result;
        using var reader = new StreamReader(stream);

        List<string> labels = new List<string>();
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            labels.Add(line);
        }
        return labels;
    }
    public class YoloOutput
    {
        private int x;

        private int y;
        private int width;
        private int height;
        private string label;
        public YoloOutput(float x, float y, float width, float height, string label)
        {
            this.x = (int)x;
            this.y = (int)y;
            this.width = (int)width;
            this.height = (int)height;

            this.label = label;
        }
    }
    private Prediction Predict(IImage originalImage) {

        // Transform Image
        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> image = 
            SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(originalImage.AsStream());

        // Preprocess image
        Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 640, 640 });
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<SixLabors.ImageSharp.PixelFormats.Rgb24> pixelSpan = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    input[0, 0, y, x] = ((pixelSpan[x].R / 255f)); 
                    input[0, 1, y, x] = ((pixelSpan[x].G / 255f));
                    input[0, 2, y, x] = ((pixelSpan[x].B / 255f));
                }
            }
        });
       
        // Setup inputs
        var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", input)
            };

        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);

        // Postprocess to get softmax vector
        var output = results.Where(x => x.Name == "output0").FirstOrDefault().AsEnumerable<float>().ToArray();
        List<YoloOutput> yoloOutputs = new List<YoloOutput>();
        for (int j = 0; j < 8400; j++)
        {
            float x = output[j];
            float y = output[j + 1];
            float width = output[j + 2];
            float height = output[j + 3];
            float[] debug = output[(j + 4)..(j + 84)];
            var (maxConf, maxConfIndex) = output[(j+4)..(j+84)].Select((f,i)=>(f,i)).Max();
            if (maxConf > _threshold * 100) {
                yoloOutputs.Add(new YoloOutput(x, y, width, height, _classNames[maxConfIndex]));
            }
            
        }
        yoloOutputs.Count();
        return new Prediction();
    }

    private async void TakePhoto(object sender, EventArgs e)
    {
        if (MediaPicker.Default.IsCaptureSupported)
        {
            FileResult photo = await MediaPicker.Default.CapturePhotoAsync();

            if (photo != null)
            {
                // Get the image
                using Stream sourceStream = await photo.OpenReadAsync();
                IImage image = PlatformImage.FromStream(sourceStream);

                // Crop the image to 224 224 pixels
                IImage newImage = image.Resize(224, 224, Microsoft.Maui.Graphics.ResizeMode.Bleed);

                // Classify the image
                Prediction prediction = Predict(newImage);

                // Show the result
                ImageCanvas.Source = ImageSource.FromStream(() => newImage.AsStream());
                Result.Text = prediction.Label;
                Accuracy.Text = $"Confidence: {prediction.Confidence*100:0.00}%";

                // Speak the result
                await TextToSpeech.Default.SpeakAsync(($"{prediction.Label}, {prediction.Confidence * 100:0.00}% sure"));
            }
        }
    }
}

