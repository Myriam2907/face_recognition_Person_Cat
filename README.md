Real-time Face Recognition System (person)
Face.py
•	To implement this code , you must install these libraries :
pip install opencv-python
pip install dlib
pip install face_recognition
RK !
•	The face_recognition library depends on the dlib library. Installing dlib can sometimes be challenging due to its dependencies.. so you can follow these steps : 
pip install cmake
install (Visual Studio Build Tools) from https://visualstudio.microsoft.com/downloads/
install (Boost) from https://www.boost.org/
pip install dlib
it should work

Test :
![1](https://github.com/Myriam2907/face_recognition_Person_Cat/assets/103574142/55ba82ff-7e23-4178-86d3-f4be24a2daec)

It detects me even in a side profile
![side](https://github.com/Myriam2907/face_recognition_Person_Cat/assets/103574142/f9b915dd-fdd6-47d0-aa4a-5b49332211ec)

It detects me even when I'm not wearing glasses
![without](https://github.com/Myriam2907/face_recognition_Person_Cat/assets/103574142/062bf147-ba41-41cd-972c-a55202206d0a)

It detects me even in a side profile without wearing glasses
![without_side](https://github.com/Myriam2907/face_recognition_Person_Cat/assets/103574142/0dd79947-d416-4ef4-a1c7-72cca62490ef)

There are no faces detected
![ghost](https://github.com/Myriam2907/face_recognition_Person_Cat/assets/103574142/d368fad6-3e96-457c-9d6c-3a74180fb588)


It detects a person's face, but it doesn't match the photo
![alert](https://github.com/Myriam2907/face_recognition_Person_Cat/assets/103574142/233b1e01-8736-4c06-893d-769a30104f4d)

Cat.py
It detects the cat's faces but does not detect human faces, puppy face, or cat with a skin care mask filter
![all](https://github.com/Myriam2907/face_recognition_Person_Cat/assets/103574142/c4bc2cc0-73b2-4a20-a4a8-222d6106ef43)







