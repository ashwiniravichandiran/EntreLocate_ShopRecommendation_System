import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { doc, collection, getDoc, setDoc } from "firebase/firestore";
import { auth, db } from "D:\\ashu_programs\\React\\Entrelocate\\src\\firebaseConfig";
import "./Home.css";
import logo from "../../assets/logo.png";

// const Home = () => {
//   const [showPopup, setShowPopup] = useState(false);
//   const [shopCategory, setShopCategory] = useState("");
//   const [ownerName, setOwnerName] = useState("");
//   const [shopAddress, setShopAddress] = useState("");
//   const [subcollection, setSubcollection] = useState(""); // To track the selected subcollection
//   const navigate = useNavigate();

//   // Check if the user already has profile data in the selected subcollection
//   const checkProfileData = async (subcollectionName) => {
//     const user = auth.currentUser;

//     if (user) {
//       const subcollectionRef = doc(
//         db,
//         "users",
//         user.email,
//         subcollectionName,
//         "profile" // Using "profile" as a placeholder document ID
//       );

//       try {
//         const docSnap = await getDoc(subcollectionRef);

//         if (docSnap.exists()) {
//           // User already has profile data in this subcollection, navigate to ProfilePage
//           const userData = docSnap.data();
//           navigate("/ProfilePage", { state: userData });
//         } else {
//           // No profile data, show the popup
//           setSubcollection(subcollectionName);
//           setShowPopup(true);
//         }
//       } catch (error) {
//         console.error("Error fetching data:", error);
//         setShowPopup(true); // In case of error, show the popup
//       }
//     } else {
//       console.log("No user is logged in.");
//     }
//   };

//   // Handle form submission for the first-time profile data
//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     const user = auth.currentUser;

//     if (user && subcollection) {
//       const subcollectionRef = doc(
//         db,
//         "users",
//         user.email,
//         subcollection,
//         "profile" // Using "profile" as the document ID within the subcollection
//       );

//       const data = { shopCategory, ownerName, shopAddress };

//       try {
//         // Save the data to Firestore under the selected subcollection
//         await setDoc(subcollectionRef, data);
//         alert("Details saved successfully!");
//         navigate("/ProfilePage", { state: data }); // Navigate to the profile page
//       } catch (error) {
//         console.error("Error saving data:", error);
//         alert("Failed to save details.");
//       }
//     } else {
//       console.log("No user is logged in or subcollection is not selected.");
//     }
//   };

//   const handleExistingBusinessClick = () => {
//     checkProfileData("existingBusinessUser");
//   };

//   const handleNewBusinessClick = () => {
//     checkProfileData("/NewBusiness");
//   };

//   return (
//     <div className="Ques1-container">
//       <nav className="containerhome">
//         <img src={logo} alt="Logo" className="logo" />
//         <ul>
//           <li>Home</li>
//           <li>Overview</li>
//           <li>Profile</li>
//         </ul>
//       </nav>
//       <div className="left-side"></div>
//       <div className="right-side">
//         <div className="right-content">
//           <h2>Select one of the below to let us know why you are here:</h2>
//         </div>
//         <div className="choice-btn-container">
//           <button
//             className="choice-btn"
//             onClick={handleNewBusinessClick} // Navigate to new business subcollection
//           >
//             I am a new Business owner and I am here looking for guidance
//           </button>
//           <button
//             className="choice-btn"
//             onClick={handleExistingBusinessClick} // Navigate to existing business subcollection
//           >
//             I am here to improve my existing business with different ideas
//           </button>
//         </div>
//       </div>

//       {showPopup && (
//         <div className="popup-overlay">
//           <div className="popup-modal">
//             <button className="close-btn" onClick={() => setShowPopup(false)}>
//               &times;
//             </button>

//             <form onSubmit={handleSubmit}>
//               <h2>Shop Details</h2>

//               <select
//                 name="shop-category"
//                 value={shopCategory}
//                 onChange={(e) => setShopCategory(e.target.value)}
//                 required
//               >
//                 <option value="" disabled>
//                   Shop Category
//                 </option>
//                 <option value="grocery">Grocery</option>
//                 <option value="clothing">Clothing</option>
//                 <option value="electronics">Electronics</option>
//                 <option value="restaurant">Restaurant</option>
//               </select>

//               <input
//                 type="text"
//                 name="owner-name"
//                 placeholder="Owner Name"
//                 value={ownerName}
//                 onChange={(e) => setOwnerName(e.target.value)}
//                 required
//               />

//               <input
//                 type="text"
//                 name="shop-address"
//                 placeholder="Shop Address"
//                 value={shopAddress}
//                 onChange={(e) => setShopAddress(e.target.value)}
//                 required
//               />

//               <button type="submit">Submit</button>
//             </form>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Home;


// import React, { useState, useEffect } from "react";
// import { useNavigate } from "react-router-dom";
// import { doc, collection, getDoc, setDoc } from "firebase/firestore";
// import { auth, db } from "D:\\react\\busi_project\\src\\firebaseConfig";
// import "./Home.css";
// import logo from "../../assets/logo.png";

const Home = () => {
  const [showPopup, setShowPopup] = useState(false);
  const [shopCategory, setShopCategory] = useState("");
  const [ownerName, setOwnerName] = useState("");
  const [shopAddress, setShopAddress] = useState("");
  const [subcollection, setSubcollection] = useState(""); // To track the selected subcollection
  const navigate = useNavigate();

  // Check if the user already has profile data in the selected subcollection
  const checkProfileData = async (subcollectionName) => {
    const user = auth.currentUser;

    if (user) {
      const subcollectionRef = doc(
        db,
        "users",
        user.email,
        subcollectionName,
        "profile" // Using "profile" as a placeholder document ID
      );

      try {
        const docSnap = await getDoc(subcollectionRef);

        if (docSnap.exists()) {
          // User already has profile data in this subcollection, navigate to ProfilePage
          const userData = docSnap.data();
          navigate("/ProfilePage", { state: userData });
        } else {
          // No profile data, show the popup
          setSubcollection(subcollectionName);
          setShowPopup(true);
        }
      } catch (error) {
        console.error("Error fetching data:", error);
        setShowPopup(true); // In case of error, show the popup
      }
    } else {
      console.log("No user is logged in.");
    }
  };

  // Handle form submission for the first-time profile data
  const handleSubmit = async (e) => {
    e.preventDefault();
    const user = auth.currentUser;

    if (user && subcollection) {
      const subcollectionRef = doc(
        db,
        "users",
        user.email,
        subcollection,
        "profile" // Using "profile" as the document ID within the subcollection
      );

      const data = { shopCategory, ownerName, shopAddress };

      try {
        // Save the data to Firestore under the selected subcollection
        await setDoc(subcollectionRef, data);
        alert("Details saved successfully!");
        navigate("/ProfilePage", { state: data }); // Navigate to the profile page
      } catch (error) {
        console.error("Error saving data:", error);
        alert("Failed to save details.");
      }
    } else {
      console.log("No user is logged in or subcollection is not selected.");
    }
  };

  const handleExistingBusinessClick = () => {
    checkProfileData("existingBusinessUser");
  };

  // const handleNewBusinessClick = () => {
  //   checkProfileData("newBusinessUser");
  // };
  const handleNewBusinessClick = () => {
    navigate("/NewBusiness"); // Directly navigate to NewBusiness.jsx
  };
  
  return (
    <div className="Ques1-container">
      <nav className="containerhome">
        <img src={logo} alt="Logo" className="logo" />
        <ul>
          <li>Home</li>
          <li>Overview</li>
          <li>Profile</li>
        </ul>
      </nav>
      <div className="left-side"></div>
      <div className="right-side">
        <div className="right-content">
          <h2>Select one of the below to let us know why you are here:</h2>
        </div>
        <div className="choice-btn-container">
          <button
            className="choice-btn"
            onClick={handleNewBusinessClick} // Navigate to new business subcollection
          >
            I am a new Business owner and I am here looking for guidance
          </button>
          <button
            className="choice-btn"
            onClick={handleExistingBusinessClick} // Navigate to existing business subcollection
          >
            I am here to improve my existing business with different ideas
          </button>
        </div>
      </div>

      {showPopup && (
        <div className="popup-overlay">
          <div className="popup-modal">
            <button className="close-btn" onClick={() => setShowPopup(false)}>
              &times;
            </button>

            <form onSubmit={handleSubmit}>
              <h2>Shop Details</h2>

              <select
                name="shop-category"
                value={shopCategory}
                onChange={(e) => setShopCategory(e.target.value)}
                required
              >
                <option value="" disabled>
                  Shop Category
                </option>
                <option value="grocery">Grocery</option>
                <option value="clothing">Clothing</option>
                <option value="electronics">Electronics</option>
                <option value="restaurant">Restaurant</option>
              </select>

              <input
                type="text"
                name="owner-name"
                placeholder="Owner Name"
                value={ownerName}
                onChange={(e) => setOwnerName(e.target.value)}
                required
              />

              <input
                type="text"
                name="shop-address"
                placeholder="Shop Address"
                value={shopAddress}
                onChange={(e) => setShopAddress(e.target.value)}
                required
              />

              <button type="submit">Submit</button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;