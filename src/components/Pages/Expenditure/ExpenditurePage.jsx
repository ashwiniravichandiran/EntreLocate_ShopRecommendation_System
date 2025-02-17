import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { db } from 'D:\\react\\busi_project\\src\\firebaseConfig';
import { doc, setDoc, getDocs,getDoc, collection, query, where } from 'firebase/firestore';



const ExpenditurePage = () => {
  const [userEmail, setUserEmail] = useState('');
  const [startMonth, setStartMonth] = useState('');
  const [endMonth, setEndMonth] = useState('');
  const [expenses, setExpenses] = useState('');
  const [outcome, setOutcome] = useState('');
  const [fixedExpenses, setFixedExpenses] = useState(0);
  const [profitOrLoss, setProfitOrLoss] = useState(null);
  const [result, setResult] = useState('');
  const [chartData, setChartData] = useState([]);

  // Fetch fixed expenses from the database
  const fetchFixedExpenses = async () => {
    if (userEmail) {
      try {
        const fixedQuery = query(collection(db, `users/${userEmail}/fixed`));
        const querySnapshot = await getDocs(fixedQuery);
        let totalFixed = 0;
        querySnapshot.forEach((doc) => {
          totalFixed += doc.data().amount || 0;
        });
        setFixedExpenses(totalFixed);
      } catch (error) {
        console.error('Error fetching fixed expenses:', error);
      }
    }
  };

  useEffect(() => {
    fetchFixedExpenses();
  }, [userEmail]);

  // Fetch existing data for the user
  useEffect(() => {
    const fetchUserData = async () => {
      if (userEmail) {
        try {
          const userQuery = query(
            collection(db, `users/${userEmail}/expenditure`)
          );
          const querySnapshot = await getDocs(userQuery);
          const data = querySnapshot.docs.map((doc) => ({
            name: `Record ${doc.id}`,
            ...doc.data(),
          }));
          const formattedData = data.map((entry) => ({
            name: `${entry.startMonth}-${entry.endMonth}`,
            value: entry.profitOrLoss,
          }));
          setChartData(formattedData);
        } catch (error) {
          console.error('Error fetching user data:', error);
        }
      }
    };

    fetchUserData();
  }, [userEmail]);

  // const calculateAndStore = async () => {
  //   const expenseValue = parseFloat(expenses);
  //   const outcomeValue = parseFloat(outcome);
  //   const totalExpenses = fixedExpenses + expenseValue;
  //   const profitOrLossValue = totalExpenses - outcomeValue;
  //   const resultValue = profitOrLossValue >= 0 ? 'Profit' : 'Loss';

  //   const data = {
  //     startMonth,
  //     endMonth,
  //     expenses: expenseValue,
  //     fixedExpenses,
  //     outcome: outcomeValue,
  //     profitOrLoss: profitOrLossValue,
  //     result: resultValue,
  //     timestamp: new Date().toISOString(),
  //   };

  //   try {
  //     const docRef = doc(
  //       db,
  //       `users/${userEmail}/expenditure`,
  //       new Date().getTime().toString()
  //     );
  //     await setDoc(docRef, data);
  //     alert('Data saved successfully!');

  //     setChartData((prevData) => [
  //       ...prevData,
  //       {
  //         name: `${startMonth}-${endMonth}`,
  //         value: profitOrLossValue,
  //       },
  //     ]);
  //   } 
  const calculateAndStore = async () => {
    const expenseValue = parseFloat(expenses) || 0;
    const outcomeValue = parseFloat(outcome) || 0;
    let fixedExpenses = 0;
  
    try {
      // Fetch fixed expenses from Firebase
      const fixedRef = doc(db, `users/${userEmail}/fixed/details`);
      const fixedDoc = await getDoc(fixedRef);
  
      if (fixedDoc.exists()) {
        const fixedData = fixedDoc.data().expenses || [];
        fixedExpenses = fixedData.reduce((sum, item) => sum + parseFloat(item.amount || 0), 0);
      }
    } catch (error) {
      console.error("Error fetching fixed expenses:", error);
    }
  
    const totalExpenses = fixedExpenses + expenseValue;
    const profitOrLossValue = outcomeValue - totalExpenses;
    const resultValue = profitOrLossValue >= 0 ? "Profit" : "Loss";
  
    const data = {
      startMonth,
      endMonth,
      expenses: expenseValue,
      fixedExpenses,
      outcome: outcomeValue,
      profitOrLoss: profitOrLossValue,
      result: resultValue,
      timestamp: new Date().toISOString(),
    };
  
    try {
      const docRef = doc(db, `users/${userEmail}/expenditure`, new Date().getTime().toString());
      await setDoc(docRef, data);
      alert("Data saved successfully!");
    } catch (error) {
      console.error("Error saving data:", error);
    }
  
    setProfitOrLoss(profitOrLossValue);
    setResult(resultValue);
  
    setChartData((prevData) => [
      ...prevData,
      { name: `${startMonth}-${endMonth}`, value: profitOrLossValue, type: "Profit/Loss" },
      { name: `${startMonth}-${endMonth}`, value: totalExpenses, type: "Expenses" },
      { name: `${startMonth}-${endMonth}`, value: outcomeValue, type: "Outcome" },
    ]);
  };
  
  return (
    <div style={styles.container}>
      <div style={styles.leftPane}>
        <h1>Expenditure Tracker</h1>
        <div style={styles.inputGroup}>
          <label>User Email:</label>
          <input
            type="email"
            value={userEmail}
            onChange={(e) => setUserEmail(e.target.value)}
            required
          />
        </div>
        <div style={styles.inputGroup}>
          <label>Start Month and Year:</label>
          <input
            type="month"
            value={startMonth}
            onChange={(e) => setStartMonth(e.target.value)}
            required
          />
        </div>
        <div style={styles.inputGroup}>
          <label>End Month and Year:</label>
          <input
            type="month"
            value={endMonth}
            onChange={(e) => setEndMonth(e.target.value)}
            required
          />
        </div>
        <div style={styles.inputGroup}>
          <label>Variable Expenses (₹):</label>
          <input
            type="number"
            value={expenses}
            onChange={(e) => setExpenses(e.target.value)}
            required
          />
        </div>
        <div style={styles.inputGroup}>
          <label>Outcome Gained (₹):</label>
          <input
            type="number"
            value={outcome}
            onChange={(e) => setOutcome(e.target.value)}
            required
          />
        </div>
        <button style={styles.button} onClick={calculateAndStore}>
          Calculate and Save
        </button>
      </div>

      <div style={styles.rightPane}>
        {result && (
          <div style={{ marginBottom: '20px' }}>
            <h2>Result: {result}</h2>
            <p>Fixed Expenses: ₹{fixedExpenses}</p>
            <p>Profit or Loss Amount: ₹{profitOrLoss}</p>
          </div>
        )}

        {chartData.length > 0 && (
          <div>
            <h3>Visual Representation</h3>
            <BarChart width={600} height={300} data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#82ca9d" name="Profit/Loss" />
              <Bar dataKey="expenses" fill="#8884d8" name="Expenses" />
              <Bar dataKey="outcome" fill="#ffc658" name="Outcome" />
            </BarChart>
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '20px',
  },
  leftPane: {
    flex: 1,
    marginRight: '20px',
  },
  rightPane: {
    flex: 1,
  },
  inputGroup: {
    marginBottom: '10px',
  },
  button: {
    marginTop: '10px',
    padding: '10px 20px',
    backgroundColor: '#4CAF50',
    color: '#fff',
    border: 'none',
    cursor: 'pointer',
  },
};

export default ExpenditurePage;
