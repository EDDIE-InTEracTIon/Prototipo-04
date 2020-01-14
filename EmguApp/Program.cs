// ------------------------------------------------------------ 
// -cristian contreras USACH DIINF ccontrerasv@gmail.com      -
// ------------------------------------------------------------

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Diagnostics;
using System.Drawing;

using HandGestureRecognition.SkinDetector;


namespace EmguHand
{
    class Program
    {
        private static Ycc YCrCb_min;
        private static Ycc YCrCb_max;
        private static IColorSkinDetector skinDetector;
        //Determina el límite de brillo al convertir la imagen en escala de grises en una imagen binaria(blanco y negro)
        private const int Threshold = 5;

        // Erosion para eliminar el ruido (reducir las zonas de píxeles blancos)

        private const int ErodeIterations = 3;

        // Dilation  para mejorar los sobrevivientes de la erosión (ampliar zonas de píxeles blancos)
        private const int DilateIterations = 3;

        //Nombres de ventanas utilizados en llamadas CvInvoke.Imshow
        private const string BackgroundFrameWindowName = "Background Frame";
        private const string RawFrameWindowName = "Raw Frame";
        private const string GrayscaleDiffFrameWindowName = "Grayscale Difference Frame";
        private const string BinaryDiffFrameWindowName = "Binary Difference Frame";
        private const string DenoisedDiffFrameWindowName = "Denoised Difference Frame";
        private const string FinalFrameWindowName = "Prototipo 4 Frame Final";

        // Contenedores para imágenes que muestran diferentes fases del procesamiento de frames
        private static Mat picture;
        private static Mat rawFrame = new Mat(); // Frame obtenido del video
        private static Mat currentFrameCopy = new Mat(); // Frame obtenido del video


        private static Mat backgroundFrame = new Mat(); //  Frame utilizado como base para la detección de cambios
        private static Mat diffFrame = new Mat(); // Imagen que muestra las diferencias entre el fondo y el marco sin formato
        private static Mat grayscaleDiffFrame = new Mat(); // Imagen que muestra diferencias de color de 8 bits
        private static Mat binaryDiffFrame = new Mat(); // Imagen que muestra áreas cambiadas en blanco y sin cambios en negro
        private static Mat denoisedDiffFrame = new Mat(); // Imagen con cambios irrelevantes eliminados con la operación de apertura
        private static Mat finalFrame = new Mat(); // Fotograma de video con el objeto detectado marcado

        private static MCvScalar drawingColor = new Bgr(Color.Red).MCvScalar;

        static void Main(string[] args)
        {
            YCrCb_min = new Ycc(0, 131, 80);
            YCrCb_max = new Ycc(255, 185, 135);
            string videoFile = @"PONER PATH DEL VIDEO";
           

            using (var capture = new VideoCapture(0)) //  Carga video del archivo
            {
                if (capture.IsOpened)
                {
                    
                    Console.WriteLine("Presione ESCAPE en cualquier imagen para salir.");
                    Console.WriteLine("Presione cualquier otra tecla para avanzar un fotograma.");

                    VideoProcessingLoop(capture, backgroundFrame);
                }
                else
                {
                    Console.WriteLine($"No se puede abrir {videoFile}");
                }
            }
        }

       private static void VideoProcessingLoop(VideoCapture capture, Mat backgroundFrame)
        {
            var stopwatch = new Stopwatch(); // Se utiliza para medir el rendimiento del procesamiento de video

            int frameNumber = 1;
            while (true) // Loop de video
            {
                rawFrame = capture.QueryFrame(); // Obteniendo el siguiente cuadro(se devuelve un valor nulo si no hay más cuadros)
                CvInvoke.Flip(rawFrame, rawFrame, FlipType.Vertical);
                CvInvoke.Flip(rawFrame, rawFrame, FlipType.Horizontal);
                Image<Bgr, Byte> currentFrameCopy = rawFrame.ToImage<Bgr, Byte>();
                picture = new Mat(@"Dr.JekyllandMr.HydeText.jpg"); //¡Elija alguna ruta en su disco!

                if (rawFrame != null) 
                {
                    frameNumber++;

                    stopwatch.Restart();
                    
                    skinDetector = new YCrCbSkinDetector();

                    Image<Gray, Byte> skin = skinDetector.DetectSkin(currentFrameCopy, YCrCb_min, YCrCb_max);
                    ProcessFrame(backgroundFrame, Threshold, ErodeIterations, DilateIterations, skin.Mat);
                    stopwatch.Stop();

                    WriteFrameInfo(stopwatch.ElapsedMilliseconds, frameNumber);
                    ShowWindowsWithImageProcessingStages();
                    CvInvoke.Imshow("Prototipo 4 Reconociento de piel", skin);

                    int key = CvInvoke.WaitKey(); // Espera indefinidamente hasta que se presione la tecla

                    //  Cierre el programa si se presionó la tecla Esc(cualquier otra tecla se mueve al siguiente cuadro)
                    if (key == 27)
                        Environment.Exit(0);
                }
                else
                {
                    capture.SetCaptureProperty(CapProp.PosFrames, 0); // Moverse al primer cuadro
                    frameNumber = 0;
                }
            }



        }

        

        private static void ProcessFrame(Mat backgroundFrame, int threshold, int erodeIterations, int dilateIterations, Mat skin)
        {
            
            rawFrame.CopyTo(finalFrame);
            DetectObject(skin, finalFrame);

           
        }

        private static void DetectObject(Mat detectionFrame, Mat displayFrame)
        {
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                VectorOfPoint biggestContour = null;
                IOutputArray hirarchy=null;
                // crea lista de contornos
                CvInvoke.FindContours(detectionFrame, contours, hirarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

                // Seleccionar el contorna mas grande
                if (contours.Size > 0)
                {
                    double maxArea = 0;
                    int chosen = 0;
                    VectorOfPoint contour = null; 
                    for (int i = 0; i < contours.Size; i++)
                    {
                         contour = contours[i];

                        double area = CvInvoke.ContourArea(contour);
                        if (area > maxArea)
                        {
                            maxArea = area;
                            chosen = i;
                        }
                    }

                    // dibuja en un frame
                    MarkDetectedObject(displayFrame, contours[chosen], maxArea);

                    VectorOfPoint hullPoints = new VectorOfPoint();
                    VectorOfInt hullInt = new VectorOfInt();

                    CvInvoke.ConvexHull(contours[chosen], hullPoints, true);
                    CvInvoke.ConvexHull(contours[chosen], hullInt, false);

                   Mat defects = new Mat();

                    if (hullInt.Size > 3)
                        CvInvoke.ConvexityDefects(contours[chosen], hullInt, defects);

                    Rectangle box = CvInvoke.BoundingRectangle(hullPoints);
                    CvInvoke.Rectangle(displayFrame, box, drawingColor);

                    Point center = new Point(box.X + box.Width / 2, box.Y + box.Height / 2);

                    VectorOfPoint start_points = new VectorOfPoint();
                    VectorOfPoint far_points = new VectorOfPoint();

                    if (!defects.IsEmpty)
                    {
                        //Los datos de Mat no se pueden leer directamente, por lo que los convertimos a Matrix<>
                        Matrix<int> m = new Matrix<int>(defects.Rows, defects.Cols,
                           defects.NumberOfChannels);
                        defects.CopyTo(m);
                        int x= 2000, y = 2000;
                        for (int i = 0; i < m.Rows; i++)
                        {
                            int startIdx = m.Data[i, 0];
                            int endIdx = m.Data[i, 1];
                            int farIdx = m.Data[i, 2];
                            Point startPoint = contours[chosen][startIdx];
                            Point endPoint = contours[chosen][endIdx];
                            Point farPoint = contours[chosen][farIdx];
                            CvInvoke.Circle(displayFrame, endPoint, 3, new MCvScalar( 0, 255,255));
                            CvInvoke.Circle(displayFrame, startPoint, 3, new MCvScalar(255, 255, 0));

                            if (true)
                            {
                                if (endPoint.Y < y)
                                {
                                    x = endPoint.X;

                                    y = endPoint.Y;

                                }

                               
                            }
                          

                            double distance = Math.Round(Math.Sqrt(Math.Pow((center.X - farPoint.X), 2) + Math.Pow((center.Y - farPoint.Y), 2)), 1);
                            if (distance<box.Height*0.3)
                            {
                                CvInvoke.Circle(displayFrame,farPoint,3, new MCvScalar(255, 0, 0));
                            }
  
                            CvInvoke.Line(displayFrame, startPoint, endPoint, new MCvScalar(0, 255, 0));

                        }
                        var info = new string[] {$"Puntero",

                $"Posicion: {x}, {y}"
            };

                        WriteMultilineText(displayFrame, info, new Point(x + 30, y));
                        CvInvoke.Circle(displayFrame, new Point(x , y), 20, new MCvScalar(255, 0, 255),2);
                        CvInvoke.Circle(picture, new Point(x*2, y*4), 20, new MCvScalar(255, 0, 255), 2);
                    }
                }
            }
        }
        

        private static void WriteFrameInfo(long elapsedMs, int frameNumber)
        {
            var info = new string[] {
                $"Numero de frame: {frameNumber}",
                $"Tiempo de procesamiento: {elapsedMs} ms"
            };

            WriteMultilineText(finalFrame, info, new Point(5, 10));
        }

        private static void ShowWindowsWithImageProcessingStages()
        {

            CvInvoke.Imshow(FinalFrameWindowName, finalFrame);
            CvInvoke.Imshow("Prototipo 4 Pagina de prueba", picture); // Abrir ventana con imagen

        }

        private static void MarkDetectedObject(Mat frame, VectorOfPoint contour, double area)
        {
            // Obteniendo un rectángulo mínimo que contiene el contorno (contour)
            Rectangle box = CvInvoke.BoundingRectangle(contour);

            // Dibujando contorno(contour) y recuadro(box) a su alrededor
            CvInvoke.Polylines(frame, contour, true, drawingColor);
            CvInvoke.Rectangle(frame, box, drawingColor);

            // Escribe información al lado del objeto marcado
                        Point center = new Point(box.X + box.Width / 2, box.Y + box.Height / 2);

            var info = new string[] {
                $"Area: {area}",
                $"Position: {center.X}, {center.Y}"
            };

            WriteMultilineText(frame, info, new Point(box.Right + 5, center.Y));
        }

        private static void WriteMultilineText(Mat frame, string[] lines, Point origin)
        {
            for (int i = 0; i < lines.Length; i++)
            {
                int y = i * 10 + origin.Y; // Baja la linea 
                CvInvoke.PutText(frame, lines[i], new Point(origin.X, y), FontFace.HersheyPlain, 0.8, drawingColor);
            }
        }
    }
}
