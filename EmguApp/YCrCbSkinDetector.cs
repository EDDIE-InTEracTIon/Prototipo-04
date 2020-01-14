using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Emgu.CV.Structure;
using Emgu.CV;
using System.Drawing;
using Emgu.CV.CvEnum;

namespace HandGestureRecognition.SkinDetector
{
    public class YCrCbSkinDetector:IColorSkinDetector
    {
        public override Image<Gray, byte> DetectSkin(Image<Bgr, byte> Img, IColor min, IColor max)
        {
            Image<Ycc, Byte> currentYCrCbFrame = Img.Convert<Ycc, Byte>();
            Image<Gray, byte> skin = new Image<Gray, byte>(Img.Width, Img.Height);
            skin = currentYCrCbFrame.InRange((Ycc)min,(Ycc) max);

            Mat rect_12 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(12, 12), new Point(6, 6));
         
            CvInvoke.Erode(skin, skin, rect_12, new Point(-1, -1), 1, BorderType.Reflect, default(MCvScalar));
            
            Mat rect_6 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(6, 6), new Point(3, 3));
        
            CvInvoke.Dilate(skin, skin, rect_6, new Point(-1, -1), 2, BorderType.Reflect, default(MCvScalar));
            return skin;
        }
        
    }
}
