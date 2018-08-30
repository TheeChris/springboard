/* Welcome to the SQL mini project. For this project, you will use
Springboard' online SQL platform, which you can log into through the
following link:

https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

Note that, if you need to, you can also download these tables locally.

In the mini project, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */



/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */
SELECT name
FROM  `Facilities` 
WHERE membercost > 0.0;

/* Q2: How many facilities do not charge a fee to members? */
SELECT COUNT( name ) 
FROM  `Facilities` 
WHERE membercost = 0.0;

/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */
SELECT facid, 
  name, 
  membercost, 
  monthlymaintenance
FROM  `Facilities` 
WHERE membercost > 0.0
AND (membercost / monthlymaintenance) < 0.2;


/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */
SELECT * 
FROM Facilities
WHERE facid IN ( 1, 5 );

/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */
SELECT name, 
  monthlymaintenance, 
  CASE WHEN monthlymaintenance >100 THEN  'expensive'
ELSE  'cheap' END AS is_expensive
FROM Facilities

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */
SELECT firstname, 
  surname AS lastname
FROM  `Members` 
ORDER BY joindate DESC 

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */
SELECT CONCAT( m.firstname,  ' ', m.surname ) AS member_name, f.name
  FROM Bookings b
  JOIN Members m ON b.memid = m.memid
  JOIN Facilities f ON b.facid = f.facid
  WHERE f.name LIKE  'Tennis%'
  GROUP BY member_name

/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */
SELECT f.name AS facility_name, 
    CONCAT( m.firstname,  ' ', m.surname ) AS member_name, 
    CASE WHEN b.memid =0 THEN f.guestcost * b.slots
    ELSE f.membercost * b.slots END AS cost
  FROM Bookings b
  JOIN Members m ON m.memid = b.memid
  JOIN Facilities f ON f.facid = b.facid
  WHERE b.starttime LIKE  '2012-09-14%'
    AND 30 < CASE WHEN b.memid =0 THEN f.guestcost * b.slots
        ELSE f.membercost * b.slots END 
  ORDER BY cost DESC 

/* Q9: This time, produce the same result as in Q8, but using a subquery. */
SELECT CONCAT( m.firstname,  ' ', m.surname ) AS member_name, 
   fac.facility_name, 
   fac.cost
  FROM (
     SELECT f.name AS facility_name, b.memid AS member_id, 
        CASE WHEN b.memid =0 THEN f.guestcost * b.slots
        ELSE f.membercost * b.slots END AS cost
     FROM Facilities f
     JOIN Bookings b ON f.facid = b.facid
     WHERE b.starttime LIKE  '2012-09-14%'
     )fac
  JOIN Members m ON m.memid = fac.member_id
  WHERE fac.cost >30
  ORDER BY fac.cost DESC

/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */
SELECT fac.facility_name, 
    SUM( fac.cost ) AS revenue
  FROM (
    SELECT f.name AS facility_name, 
       b.memid AS member_id, 
       CASE WHEN b.memid =0 THEN f.guestcost * b.slots
       ELSE f.membercost * b.slots END AS cost
    FROM Facilities f
    JOIN Bookings b ON f.facid = b.facid
    )fac
   JOIN Members m ON m.memid = fac.member_id
   GROUP BY facility_name
   HAVING revenue <1000
   ORDER BY revenue DESC 
