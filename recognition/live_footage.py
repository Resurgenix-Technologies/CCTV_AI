import cv2
import cv2
import NDIlib as ndi

if not ndi.initialize():
    raise RuntimeError("NDI init failed")

finder = ndi.find_create_v2()
ndi.find_wait_for_sources(finder, 5000)

sources = ndi.find_get_current_sources(finder)

recv = ndi.recv_create_v3()
ndi.recv_connect(recv, sources[0])

print("Connected:", sources[0].ndi_name)

while True:

    frame_type, video_frame, _, _ = ndi.recv_capture_v2(recv, 1000)

    if frame_type == ndi.FRAME_TYPE_VIDEO:

        # UYVY -> BGR
        frame = cv2.cvtColor(video_frame.data, cv2.COLOR_YUV2BGR_UYVY)

        cv2.imshow("NDI Live", frame)

        ndi.recv_free_video_v2(recv, video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
ndi.recv_destroy(recv)
ndi.destroy()