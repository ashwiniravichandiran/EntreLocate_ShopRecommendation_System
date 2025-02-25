import { getAuth, GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
import React from 'react';
import googleLogo from 'D:\\ashu_programs\\React\\Entrelocate\\src\\assets\\google2.png';
import { app } from 'D:\\ashu_programs\\React\\Entrelocate\\src\\components\\firebase\\firebase'; // Import app as a named export

const auth = getAuth(app);

function googleLogin() {
  const provider = new GoogleAuthProvider();
  signInWithPopup(auth, provider).then((result) => {
    console.log(result);
  }).catch((error) => {
    console.error("Error during sign-in:", error);
  });
}

const Signinwithgoogle = () => {
  return (
    <div>
      <div
        style={{ display: "flex", justifyContent: "center", cursor: "pointer" }}
        onClick={googleLogin}>
        <img src={googleLogo} alt="Google sign-in" width="60%" />
      </div>
    </div>
  );
};

export default Signinwithgoogle;