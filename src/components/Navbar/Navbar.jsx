import React from 'react';
import './Navbar.css';
import logo from '../../assets/logo.png';

const Navbar = ({ onLoginClick }) => {
  return (
    <nav className='container'>
      <img src={logo} alt="" className='logo'/>
      <ul>
        <li>Overview</li>
        <li><button className='btn' onClick={onLoginClick}>Login</button></li>
      </ul>
    </nav>
  );
};

export default Navbar;