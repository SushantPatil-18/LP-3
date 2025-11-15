// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract Bank {
    mapping(address => uint256) private balances;

    event Deposited(address indexed who, uint256 amount);
    event Withdrawn(address indexed who, uint256 amount);

    // deposit payable - increases sender balance
    function deposit() external payable {
        require(msg.value > 0, "zero deposit");
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    // withdraw up to sender's balance
    function withdraw(uint256 amount) external {
        require(amount > 0, "zero withdraw");
        uint256 bal = balances[msg.sender];
        require(bal >= amount, "insufficient balance");
        balances[msg.sender] = bal - amount;
        payable(msg.sender).transfer(amount);
        emit Withdrawn(msg.sender, amount);
    }

    // view balance of caller
    function balance() external view returns (uint256) {
        return balances[msg.sender];
    }

    // optional: view balance of any account
    function balanceOf(address acct) external view returns (uint256) {
        return balances[acct];
    }
}
