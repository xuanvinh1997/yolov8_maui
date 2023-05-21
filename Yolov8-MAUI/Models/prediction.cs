using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Yolov8_MAUI.Models
{
    public class Prediction
    {
        public string Label { get; set; }

        public float Confidence { get; set; }
    }

}
