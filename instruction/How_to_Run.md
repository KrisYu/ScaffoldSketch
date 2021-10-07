# How to Run


## On Mac/PC

- `python3 app.py`

- Once it launches, you will see a small window like this:

![](images/app_interface.png)


The window will display an address similar to above, type `https://192.168.0.195:4443`  (replace this with yours) to the Oculus Quest Browser.

### On Oculus Quest


- Type the above address in the Oculus Quest Browser. You might need to used advanced button to accpet the warning and continue.
- Press 'Enter VR'.

![](images/html.png)

![](images/controllers.png)

In VR, the controller buttons have text labels displayed whenever you bring the controller close to your eyes.

![](images/right_controller.png)
![](images/left_controller.png)

The system trying to mimic how product designers sketch with pen and paper, it provides 2 different drawing modes - scaffold/shape. Initially, it's in scaffold mode. You can easily tell the drawing modes by whether there's a straight line or curve near the drawing tip, or bring the controller close to you.

![](images/right_controller_mode.png)

In the scaffold mode, you can draw straight lines and attach points to the lines.

In the shape mode, you can draw nice shape curves.


**Depend on the system, the below results might differ, you might see a system beautify the curves you draw or not.**

### Your first drawing

Let's begin drawing a rounded corner square.

First, let's draw a scaffold square

- Draw the 1st line.

![](images/1st_line.png)

You'll notice as a line generated, there are 3 points also marked on the line, that's the endpoints and midpoint.

- Draw the 2nd line.

![](images/2nd_line.png)

You can see that your 2nd line's endpoints and direction and length are snapped with respect to the 1st line. People in VR cannot draw as accurately as in 2D. The scaffold mode will do automatic snapping for you.

- Draw the 3rd and 4th lines.

![](images/4th_line.png)


Press A to switch to shape mode, try to draw a circle in the square.

If you succeed, great! The shape mode works by detecting the scaffold lines and points on the scaffold lines and the curve you draw, in combination it beautifies the curve.

If you failed, it might be the curve is not near to the scaffold. As mentioned we use the combination to beautify the curve. Only the curve and the scaffold points are close enough it will activate the beautification, otherwise, you might get different shape curves or even no curve at all. You can push the right controller's thumbstick left to undo and clean the circle.

Press A again to switch back to scaffold mode. Now let's try the rounded corner square. 

If you try to draw a very short line (or just press the draw) near the scaffold line, it will snap a point on the nearest scaffold line. 

Draw 2 points to mark the start and end of one rouned corner.

![](images/one_corner.png)

Press A again to shape mode to draw the rounded corner. 

You might succeed or fail in this step. 

As mentioned, the system detects scaffold points/lines and the curve you draw in combination to beautify the curve. So it might also detect the corner point if these 3 points are close. When there are many key points crowded and you want to draw precisely. You can use the zoom out/zoom in function.

Place the left controller to where you want to zoom out, then press the controller's thumbstick to the right, you'll see the shape zoomed out. And the sphere on the left controller(as a zoom indicator)  also zoomed. Try again now.

![](images/zoom_1st_corner.png)

Now do the same and draw the other corner.

![](images/2nd_corner.png)

In the shape mode, connect the 2 corners.

![](images/one_side.png)

Now you can continue and finish the rounded corner square.

Anytime in the drawing process, you can use undo, or redo if you regret the undo. And you can zoom out/zoom in.

After you finished the rounded corner square, you can hide the scaffold and view your first rounded corner square in VR.

Press X to export. This step will create a VR_output folder on Desktop (if the folder not exists) and create a time_stamp.gltf file and a time_stamp.json file in the folder.

Great! You have finished your tutorial, you have learned how to switch mode, undo/redo, zoom out/zoom in, hide/show scaffold, and export.


### The exported file

The exported gltf file can be viewed using [https://gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com)

Make sure you check the wireframe, you can also check the background for a better view effect.

![](images/gltf_view.png) 



