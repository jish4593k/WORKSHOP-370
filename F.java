import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.core.CvType;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;

public class SmileDetection {

    private static Net net;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Smile Detection");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(800, 600);

            JLabel label = new JLabel();
            frame.add(label);

            frame.setVisible(true);

            net = Dnn.readNetFromTensorflow("assets/graph_opt.pb");

            VideoCapture camera = new VideoCapture(0);
            camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 800);
            camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 600);

            Mat frameMat = new Mat();

            while (true) {
                if (camera.read(frameMat)) {
                    detectAndDisplay(frameMat, label);
                } else {
                    System.out.println("Error: Cannot read frame.");
                    break;
                }
            }

            camera.release();
        });
    }

    private static void detectAndDisplay(Mat frame, JLabel label) {
        MatOfFloat confidences = new MatOfFloat();
        MatOfInt indices = new MatOfInt();
        Mat blob = Dnn.blobFromImage(frame, 1.0, new Size(300, 300), new Scalar(127.5, 127.5, 127.5), true, false, CvType.CV_32F);
        net.setInput(blob);
        net.forward(confidences, indices);

        int rows = indices.rows();
        for (int i = 0; i < rows; ++i) {
            int index = (int) indices.get(i, 0)[0];
            float confidence = confidences.get(i, 0)[0];

            if (confidence > 0.5) {
                float x = frame.cols() * blob.get(0, 0, 0)[index];
                float y = frame.rows() * blob.get(0, 0, 1)[index];
                float width = frame.cols() * blob.get(0, 0, 2)[index];
                float height = frame.rows() * blob.get(0, 0, 3)[index];

                // Draw a rectangle around the detected face
                Imgproc.rectangle(frame, new Point(x, y), new Point(x + width, y + height), new Scalar(0, 255, 0), 2);
                Imgproc.putText(frame, "Smiling!", new Point(x, y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 255, 0), 2);
            }
        }

        Image image = mat2Image(frame);
        label.setIcon(new ImageIcon(image));
    }

    private static Image mat2Image(Mat frame) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", frame, buffer);

        byte[] data = buffer.toArray();
        BufferedImage bufImage = new BufferedImage(frame.cols(), frame.rows(), BufferedImage.TYPE_3BYTE_BGR);

        bufImage.getRaster().setDataElements(0, 0, frame.cols(), frame.rows(), data);
        return bufImage;
    }
}
