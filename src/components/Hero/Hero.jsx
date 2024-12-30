


// import React from 'react';
// import './Hero.css';
// import dark_arrow from '../../assets/dark-arrow.png';
// import { useNavigate } from "react-router-dom";

// const Hero = ({ isVisible }) => {
//   const navigate = useNavigate();
//   return (
//     <div className='hero container'>
//       {isVisible && (
//         <div className='hero-text'>
//           <h1>Explore the best place to find your Business!</h1>
//           <p>Get started with us</p>
//           <button className='btn' onClick={() => navigate("/signup")}>  {/* Update to onClick */}
//             Sign up <img src={dark_arrow} alt=""/>
//           </button>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Hero;


// import React from 'react';
// import './Hero.css';
// import dark_arrow from '../../assets/dark-arrow.png';
// import { useNavigate } from "react-router-dom";
// import heroVideo from 'D:\\react\\busi_project\\src\\assets\\hero1.mp4'; // Make sure to place the video in the assets folder

// const Hero = ({ isVisible }) => {
//   const navigate = useNavigate();
  
//   return (
//     <div className='hero'>
//       <video className="hero-video" autoPlay loop muted>
//         <source src={heroVideo} type="video/mp4" />
//         Your browser does not support the video tag.
//       </video>

//       {isVisible && (
//         <div className='hero-text'>
//           <h1>Explore the best place to find your Business!</h1>
//           <p>Get started with us</p>
//           <button className='btn' onClick={() => navigate("/signup")}>
//             Sign up <img src={dark_arrow} alt=""/>
//           </button>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Hero;

import React from 'react';
import './Hero.css';
import dark_arrow from '../../assets/dark-arrow.png';
import heroVideo from '../../assets/hero1.mp4'; // Import the video file
import { useNavigate } from "react-router-dom";

const Hero = ({ isVisible }) => {
  const navigate = useNavigate();
  
  return (
    <div className='hero container'>
      {/* Background Video */}
      <video className='hero-video' autoPlay muted loop>
        <source src={heroVideo} type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      {isVisible && (
        <div className='hero-text'>
          <h1>Explore the best place to find your Business!</h1>
          <p>Get started with us</p>
          <button className='btn' onClick={() => navigate("/signup")}>
            Sign up <img src={dark_arrow} alt=""/>
          </button>
        </div>
      )}
    </div>
  );
};

export default Hero;
