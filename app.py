import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import io

## INFRASTRUCTURE
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

DEMO_IMAGE = './content/sample.jpg'
SHOWCASE_IMAGE = './content/showcase.jpg'

st.title("Pose Estimation with MediaPipe")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false] > div:first-child{
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Pose Estimation Sidebar')
st.sidebar.subheader('Parameters')


## Caching
@st.cache()
## Two Cases: Images and Videos
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        #r = width / float(w)
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    ## Resize Image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


## Various Application States
app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App', 'Run on Image', 'Run on Video']
                                )

if app_mode == 'About App':
    st.markdown("An application of Machine Learning to estimate pose in a served image, "
                "modeled on the MediaPipe infrastructure, method is fast and computational cheap. "
                "\n\n"
                "To use the app, follow the sidebar and enter an image of choice and play around."
                "\n\n\n"
                "Project is heavily influenced by the work of [Ritesh Kanjee|Augmented StartUps](https://www.augmentedstartups.com/)")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false] > div:first-child{
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.image(SHOWCASE_IMAGE)

elif app_mode == 'Run on Image':

    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:  # default to demo image
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
    ) as pose:
        ## Convert BGR -> RGB
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        ## Draw pose
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)

        st.subheader('Output Image')
        st.image(annotated_image, use_column_width=True)