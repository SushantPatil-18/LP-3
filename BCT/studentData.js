const { ethers } = require("hardhat");

async function main() {
  const Student = await ethers.getContractFactory("StudentRegistry");
  const student = await Student.deploy();
  await student.deployed();
  console.log("Deployed to:", student.address);

  // Add a student and measure gas
  const tx = await student.addStudent(1, "Alice", 21);
  const receipt = await tx.wait();

  const gasUsed = receipt.gasUsed; // BigNumber
  const effectiveGasPrice = receipt.effectiveGasPrice; // BigNumber (wei)

  const feeWei = gasUsed.mul(effectiveGasPrice);
  console.log("gasUsed:", gasUsed.toString());
  console.log("effectiveGasPrice (wei):", effectiveGasPrice.toString());
  console.log("tx fee (wei):", feeWei.toString());
  console.log("tx fee (ETH):", ethers.utils.formatEther(feeWei));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});


// Deploy & inspect gas programmatically (Hardhat + Ethers.js)

// If you prefer to deploy with Hardhat and print gas usage:

// Install Hardhat and ethers: npm init -y && npm install --save-dev hardhat then npm install ethers

// Example deploy script scripts/deploy.js:

// Run with npx hardhat run --network <network> scripts/deploy.js (or use --network localhost for local node). The script prints gasUsed, effectiveGasPrice, and tx fee in ETH.Run with npx hardhat run --network <network> scripts/deploy.js (or use --network localhost for local node). The script prints gasUsed, effectiveGasPrice, and tx fee in ETH.