// scripts/deploy.js
async function main() {
  const [deployer] = await ethers.getSigners();
  const Bank = await ethers.getContractFactory("Bank");
  const bank = await Bank.deploy();
  await bank.deployed();
  console.log("Bank deployed to:", bank.address);
}
main().catch((err) => { console.error(err); process.exitCode = 1; });


// Run deploy (example for Sepolia/testnet configured in hardhat.config.js):

// npx hardhat run --network <your-testnet> scripts/deploy.js