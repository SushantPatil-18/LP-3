// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title Student Registry (example demonstrating struct, array, fallback/receive)
/// @notice Simple contract to add/list students, accepts ETH via receive/fallback, emits events
contract StudentRegistry {
    // --- Structures ---
    struct Student {
        uint256 id;
        string name;
        uint8 age;
        address addedBy;
        uint256 timestamp;
    }

    // --- Storage ---
    Student[] private students;                // dynamic array of students
    mapping(uint256 => uint256) private idxOf; // maps id -> index+1 (0 means not found)

    address public owner;

    // --- Events ---
    event StudentAdded(uint256 indexed id, string name, uint8 age, address indexed addedBy);
    event StudentRemoved(uint256 indexed id, address indexed removedBy);
    event Received(address indexed from, uint256 amount, string note);
    event FallbackCalled(address indexed from, uint256 amount, string note);

    // --- Modifiers ---
    modifier onlyOwner() {
        require(msg.sender == owner, "only owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    // --- Functions using struct & array ---

    /// @notice Add a new student. Reverts if ID already exists.
    function addStudent(uint256 _id, string calldata _name, uint8 _age) external {
        require(_id != 0, "id cannot be zero");
        require(idxOf[_id] == 0, "id exists");

        Student memory s = Student({
            id: _id,
            name: _name,
            age: _age,
            addedBy: msg.sender,
            timestamp: block.timestamp
        });

        students.push(s);
        idxOf[_id] = students.length; // store index+1

        emit StudentAdded(_id, _name, _age, msg.sender);
    }

    /// @notice Get student by id. Reverts if not found.
    function getStudent(uint256 _id) external view returns (Student memory) {
        uint256 i = idxOf[_id];
        require(i != 0, "not found");
        return students[i - 1];
    }

    /// @notice Return number of students
    function studentCount() external view returns (uint256) {
        return students.length;
    }

    /// @notice Return all students (be careful with very large arrays — may revert due to gas limits)
    function getAllStudents() external view returns (Student[] memory) {
        return students;
    }

    /// @notice Remove a student by id (swaps with last element to keep array dense)
    function removeStudent(uint256 _id) external onlyOwner {
        uint256 i = idxOf[_id];
        require(i != 0, "not found");
        uint256 index = i - 1;
        uint256 lastIndex = students.length - 1;

        if (index != lastIndex) {
            // swap
            Student memory lastStudent = students[lastIndex];
            students[index] = lastStudent;
            idxOf[lastStudent.id] = index + 1;
        }

        students.pop();
        delete idxOf[_id];

        emit StudentRemoved(_id, msg.sender);
    }

    // --- Payable & fallback features ---

    /// @notice Allow direct ETH transfers to the contract and emit Received
    receive() external payable {
        // Called when msg.data is empty
        emit Received(msg.sender, msg.value, "receive()");
    }

    /// @notice fallback triggered when msg.data not empty or no matching function signature.
    fallback() external payable {
        // fallback can be used to handle unexpected calls
        emit FallbackCalled(msg.sender, msg.value, "fallback()");
    }

    /// @notice Owner can withdraw Ether accumulated in contract.
    function withdraw(address payable _to, uint256 _amount) external onlyOwner {
        require(address(this).balance >= _amount, "insufficient balance");
        _to.transfer(_amount);
    }

    /// @notice View contract balance
    function contractBalance() external view returns (uint256) {
        return address(this).balance;
    }
}
