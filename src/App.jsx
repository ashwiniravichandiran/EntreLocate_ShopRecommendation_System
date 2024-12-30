import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import Hero from './components/Hero/Hero';
import LoginBox from './components/LoginBox/LoginBox';
import Signup from './components/Pages/Signup/Signup'; // Adjust the path if needed
import Home from './components/Home/Home';  // Adjust the path if needed
import NewBusiness from './components/Pages/NewBusiness/NewBusiness'; 
import Location_2 from './components/Pages/NewBusiness/location_2';
// import ClusterDetail from './components/Pages/NewBusiness/ClusterDetail';
// import Popup from './components/Pages/NewBusiness/Popup';

const App = () => {
  const [isHeroContentVisible, setHeroContentVisible] = useState(true);
  const [showLoginBox, setShowLoginBox] = useState(false);

  const handleLoginClick = () => {
    setHeroContentVisible(false);
    setShowLoginBox(true);
  };

  const handleCloseLoginBox = () => {
    setHeroContentVisible(true); // Show the Hero content again
    setShowLoginBox(false);
  };

  return (
    <Router>
      <AppWithNavbar
        handleLoginClick={handleLoginClick}
        handleCloseLoginBox={handleCloseLoginBox}
        isHeroContentVisible={isHeroContentVisible}
        showLoginBox={showLoginBox}
      />
    </Router>
  );
};

const AppWithNavbar = ({ handleLoginClick, handleCloseLoginBox, isHeroContentVisible, showLoginBox }) => {
  const location = useLocation(); // useLocation must be inside the Router component

  return (
    <div>
      {location.pathname === '/' && <Navbar onLoginClick={handleLoginClick} />}
      <Routes>
        <Route path="/" element={<Hero isVisible={isHeroContentVisible} />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/home" element={<Home />} /> {/* Home page route */}
        <Route path="/login" element={
          showLoginBox ? <LoginBox onClose={handleCloseLoginBox} /> : <Hero isVisible={isHeroContentVisible} />
        } />
         <Route path="/NewBusiness" element={<NewBusiness />} />
         {/* <Route path="/Popup" element={<Popup />} /> */}
         <Route path="/location_2" element={<Location_2 />} />
         
      </Routes>
    
      {showLoginBox && <LoginBox onClose={handleCloseLoginBox} />}
    </div>
  );
};

export default App;

// import React, { useState } from 'react';
// import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
// import Navbar from './components/Navbar/Navbar';
// import Hero from './components/Hero/Hero';
// import LoginBox from './components/LoginBox/LoginBox';
// import Signup from 'D:\react\busi_project\src\components\Pages\Signup\Signup'; // Adjust the path if needed
// import Home from './components/Home/Home';  // Adjust the path if needed
// import NewBusiness from 'D:\react\busi_project\src\components\Pages\NewBusiness\NewBusiness';


// const App = () => {
//   const [isHeroContentVisible, setHeroContentVisible] = useState(true);
//   const [showLoginBox, setShowLoginBox] = useState(false);
  

//   const handleLoginClick = () => {
//     setHeroContentVisible(false);
//     setShowLoginBox(true);
//   };

//   const handleCloseLoginBox = () => {
//     setHeroContentVisible(true); // Show the Hero content again
//     setShowLoginBox(false);
//   };

//   return (
//     <Router>
//       <AppWithNavbar
//         handleLoginClick={handleLoginClick}
//         handleCloseLoginBox={handleCloseLoginBox}
//         isHeroContentVisible={isHeroContentVisible}
//         showLoginBox={showLoginBox}
//       />
//     </Router>
//   );
// };

// const AppWithNavbar = ({ handleLoginClick, handleCloseLoginBox, isHeroContentVisible, showLoginBox }) => {
//   const location = useLocation(); // useLocation must be inside the Router component

//   return (
//     <div>
//       {location.pathname === '/' && <Navbar onLoginClick={handleLoginClick} />}
//       <Routes>
//         <Route path="/" element={<Hero isVisible={isHeroContentVisible} />} />
//         <Route path="/signup" element={<Signup />} />
//         <Route path="/home" element={<Home />} />
//         <Route path="/NewBusiness" element={<NewBusiness />} />
//         {/* <Route path="/Location" element={<Location />} /> */}
//         <Route path="/login" element={
//           showLoginBox ? <LoginBox onClose={handleCloseLoginBox} /> : <Hero isVisible={isHeroContentVisible} />
//         } />
//       </Routes>
//       {showLoginBox && <LoginBox onClose={handleCloseLoginBox} />}
//     </div>
//   );
// };

// export default App;