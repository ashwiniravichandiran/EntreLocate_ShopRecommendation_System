import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { db, storage, auth } from "D:\\react\\busi_project\\src\\firebaseConfig";
import { doc, getDoc, updateDoc, setDoc } from "firebase/firestore";
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import "./ProfilePage.css";

const ProfilePage = () => {
  const [shopCategory, setShopCategory] = useState("");
  const [ownerName, setOwnerName] = useState("");
  const [shopAddress, setShopAddress] = useState("");
  const [uploadedPhoto, setUploadedPhoto] = useState(null);
  const [photoFile, setPhotoFile] = useState(false); // For the photo upload
  const [isEditing, setIsEditing] = useState(false);
  const [fixedExpenses, setFixedExpenses] = useState([]);
  const [popupVisible, setPopupVisible] = useState(false); // Popup visibility
  const [newExpenses, setNewExpenses] = useState([{ name: "", amount: "" }]);

  const navigate = useNavigate();
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(true);

  // Handle photo upload
  const handlePhotoUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setPhotoFile(file);
      const photoRef = ref(storage, `users/${auth.currentUser.email}/photo.jpg`);
      await uploadBytes(photoRef, file);
      const photoURL = await getDownloadURL(photoRef);
      setUploadedPhoto(photoURL);
      alert("Photo uploaded successfully!");
    }
  };

  // Fetch data on component load
  useEffect(() => {
    const fetchUserData = async () => {
      const user = auth.currentUser;
      if (user) {
        try {
          const userRef = doc(db, "users", user.email);
          const docSnap = await getDoc(userRef);
          if (docSnap.exists()) {
            const data = docSnap.data();
            setShopCategory(data.shopCategory || "");
            setOwnerName(data.ownerName || "");
            setShopAddress(data.shopAddress || "");
            setFixedExpenses(data.fixedExpenses || []);
            const photoRef = ref(storage, `users/${user.email}/photo.jpg`);
            try {
              const photoURL = await getDownloadURL(photoRef);
              setUploadedPhoto(photoURL);
            } catch {
              console.log("No photo found.");
            }
          }
        } catch (error) {
          console.error("Error fetching user data:", error);
        }
      }
      setIsLoading(false);
    };

    if (!location.state) {
      fetchUserData();
    } else {
      const { shopCategory, ownerName, shopAddress, fixedExpenses } = location.state;
      setShopCategory(shopCategory);
      setOwnerName(ownerName);
      setShopAddress(shopAddress);
      setFixedExpenses(fixedExpenses || []);
      setIsLoading(false);
    }
  }, [location.state]);

  // Handle popup save
  const handleSaveExpenses = async () => {
    const userRef = doc(db, "users", auth.currentUser.email, "fixed", "details");
    try {
      await setDoc(userRef, { expenses: newExpenses }, { merge: true });
      alert("Expenses saved successfully!");
      setPopupVisible(false);
    } catch (error) {
      console.error("Error saving expenses:", error);
    }
  };

  // Add new row
  const addNewExpenseRow = () => {
    setNewExpenses([...newExpenses, { name: "", amount: "" }]);
  };

  // Handle expenditure button click
  const handleExpenditureClick = () => {
    navigate("/expenditure");
  };

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="profile-container">
      {/* Left Section */}
      <div className="left-section">
        <div className="photo-upload">
          {uploadedPhoto ? (
            <img src={uploadedPhoto} alt="Uploaded" className="uploaded-photo" />
          ) : (
            <div className="photo-placeholder">No Photo</div>
          )}
          <label className="upload-label">
            Upload Photo
            <input
              type="file"
              className="upload-input"
              onChange={handlePhotoUpload}
              accept="image/*"
            />
          </label>
        </div>

        <div className="button-group">
          <button
            className="action-btn"
            onClick={() => setPopupVisible(true)} // Show popup
          >
            FIXED
          </button>
          {/*<button className="action-btn" onClick={handleExpenditureClick}>
            EXPENDITURE
          </button> */}
          <button
  className="action-btn"
  onClick={() => navigate("/expenditure")} // Match the route in App.jsx
>
  EXPENDITURE
</button>

        </div>
      </div>

      {/* Right Section */}
      <div className="right-section">
        <div className="profile-card">
          <h2>Owner and Shop Details</h2>
          <div>
            <strong>Shop Category:</strong> {shopCategory}
          </div>
          <div>
            <strong>Owner Name:</strong> {ownerName}
          </div>
          <div>
            <strong>Shop Address:</strong> {shopAddress}
          </div>
        </div>
      </div>

      {/* Fixed Expenses Popup */}
      {popupVisible && (
        <div className="fixed-popup-overlay">
          <div className="fixed-popup">
            <h2>Fixed Expenses</h2>
            {newExpenses.map((expense, index) => (
              <div key={index} className="add-expense">
                <input
                  type="text"
                  placeholder="Name"
                  value={expense.name}
                  onChange={(e) =>
                    setNewExpenses(
                      newExpenses.map((exp, i) =>
                        i === index ? { ...exp, name: e.target.value } : exp
                      )
                    )
                  }
                />
                <input
                  type="number"
                  placeholder="Amount"
                  value={expense.amount}
                  onChange={(e) =>
                    setNewExpenses(
                      newExpenses.map((exp, i) =>
                        i === index ? { ...exp, amount: e.target.value } : exp
                      )
                    )
                  }
                />
              </div>
            ))}
            <button onClick={addNewExpenseRow}>Add</button>
            <button onClick={handleSaveExpenses}>Save</button>
            <button className="close-btn" onClick={() => setPopupVisible(false)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfilePage;
