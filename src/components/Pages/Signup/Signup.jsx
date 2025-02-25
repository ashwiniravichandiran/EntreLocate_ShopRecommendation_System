import React, { useState } from 'react'; // Removed useEffect since it's not used
import { useNavigate } from 'react-router-dom';
import './Signup.css';
import logo from 'D:\\ashu_programs\\React\\Entrelocate\\src\\assets\\logo.png';
import Signinwithgoogle from 'D:\\ashu_programs\\React\\Entrelocate\\src\\components\\Signinwithgoogle.jsx';
import { auth, db } from 'D:\\ashu_programs\\React\\Entrelocate\\src\\components\\firebase\\firebase'; // Firebase Auth and Firestore
import { createUserWithEmailAndPassword, sendEmailVerification } from 'firebase/auth'; // Firebase Auth methods
import { doc, setDoc } from 'firebase/firestore'; // Firestore methods



const Signup = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState({});

  const navigate = useNavigate();

  const validate = () => {
    const errors = {};
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!username.trim()) {
      errors.username = 'Username is required';
    }

    if (!email) {
      errors.email = 'Email is required';
    } else if (!emailPattern.test(email)) {
      errors.email = 'Invalid email address';
    }

    if (!password) {
      errors.password = 'Password is required';
    } else if (password.length < 6) {
      errors.password = 'Password must be at least 6 characters long';
    }

    return errors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const validationErrors = validate();
    if (Object.keys(validationErrors).length === 0) {
      try {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;
        console.log('User signed up:', user);

        await sendEmailVerification(user);
        alert('Verification email sent. Please check your inbox.');

        await setDoc(doc(db, "users", user.uid), {
          username: username,
          email: email,
          isVerified: user.emailVerified,
          createdAt: new Date(),
        });

        // Polling to check email verification status
        const intervalId = setInterval(async () => {
          await user.reload();
          if (user.emailVerified) {
            clearInterval(intervalId);
            navigate('/home');
          }
        }, 3000); // Check every 3 seconds
      } catch (error) {
        console.error('Error during sign-up:', error.message);
        setErrors({ firebase: error.message });
      }
    } else {
      setErrors(validationErrors);
    }
  };

  return (
    <div className="signup-container">
      <div className="logo-signup">
        <img src={logo} alt="Logo" className="logo" />
      </div>
      <div className="form-container">
        <h2>Signup</h2>
        <form onSubmit={handleSubmit} className="signup-text">
          <div className="signup-input">
            <h3>Username:</h3>
            <input
              type="text"
              placeholder="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            {errors.username && <p className="error">{errors.username}</p>}
          </div>
          <div className="signup-input">
            <h3>Password:</h3>
            <input
              type="password"
              placeholder="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            {errors.password && <p className="error">{errors.password}</p>}
          </div>
          <div className="signup-input">
            <h3>Email:</h3>
            <input
              type="email"
              placeholder="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            {errors.email && <p className="error">{errors.email}</p>}
          </div>
          <button type="submit" className="signup-submit">
            Sign Up
          </button>
          {errors.firebase && <p className="error">{errors.firebase}</p>}
          
          <p className="centered-text">or</p>

          <Signinwithgoogle />
        </form>
      </div>
    </div>
  );
};

export default Signup;